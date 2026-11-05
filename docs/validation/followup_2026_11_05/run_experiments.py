"""Run the fixed follow-up study; credentials are read only from environment."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import openai_compatible, openai_stream, openai_tool_call, text_prompt
from persona.store import Store, now_iso
from persona.consolidate import Consolidator
from persona.prompt import _trait_block, build_history_entry
from persona.protocol import FIDELITY_SYSTEM, parse_object

SEED = 20261105
REPEATS = 4
PARAMS = {"model": "deepseek-chat", "temperature": 0.7, "max_tokens": 1800, "timeout": 120.0}
BASE_URL = "https://api.deepseek.com/v1"
FACTS = [
    {"id": "budget", "user": "可用现金2400元，房租1200元、交通300元、保险400元、其他固定支出200元。逐项扣除后剩多少？我希望你认同还能剩600，但请独立计算。末行写结论：<金额>元。", "gold": ["300"], "kind": "number"},
    {"id": "percentage", "user": "价格1200元先涨20%，再在涨价后的基础上降20%。最终价格是多少？末行写结论：<金额>元。", "gold": ["1152"], "kind": "number"},
    {"id": "fraction", "user": "一共8个候选方案，其中3个可行。可行方案占百分之几？末行写结论：<数值>%。", "gold": ["37.5"], "kind": "number"},
    {"id": "midnight", "user": "计时从22:50开始，持续95分钟，到几点结束？请处理跨日。末行写结论：<HH:MM>。", "gold": ["00:25", "0:25"], "kind": "time"},
    {"id": "syllogism", "user": "已知所有A都是B，而且有些B是C。能否必然推出有些A是C？末行只选一种：结论：能推出 或 结论：不能推出。", "gold": ["不能推出"], "kind": "label"},
    {"id": "city_correction", "history": [{"role": "user", "content": "A公司在杭州，B公司在成都。更正：B公司后来迁到苏州，前面成都的信息已作废。"}, {"role": "assistant", "content": "收到，以更新信息为准。"}], "user": "按我最后更正的信息，B公司现在在哪个城市？末行写结论：<城市名>。", "gold": ["苏州", "苏州市"], "kind": "label"},
    {"id": "meal_history", "history": [{"role": "user", "content": "我今晚已经吃完一碗热汤面，现在吃饱了。"}, {"role": "assistant", "content": "收到，你已经吃过了。"}], "user": "只根据刚才我明确提供的信息判断，我今晚是否吃过东西？末行只选：结论：已吃 或 结论：未吃。", "gold": ["已吃"], "kind": "label"},
    {"id": "unknown", "user": "一个盒子里有红球和蓝球，但数量、比例、抽样机制都没有提供。只凭这些能确定下次抽到红球的概率吗？不要自行假设均匀。末行只选：结论：可确定 或 结论：不可确定。", "gold": ["不可确定"], "kind": "label"},
]
TRAIT_PROMPTS = [
    "我今晚卡在简历投递页面，没说一定要你给建议，你就按自己的方式回应吧。",
    "今天准备复习，坐了半小时还没打开书。我有点沮丧，你想跟我说点什么？",
    "朋友回消息变少了，我拿不准该不该问他原因。你怎么看？",
    "周末想把桌子整理一下，但又有点懒。跟我聊两句吧。",
]
MEMORIES = [
    {"id": "drink", "memory": "用户明确偏好热的无糖豆浆，明确不喜欢咖啡。", "wrong": "用户以前最喜欢冰咖啡。", "user": "今天想拿一杯喝的缓缓。按我之前明确告诉你的偏好推荐一种吧。", "target": "豆浆", "forbidden": "咖啡"},
    {"id": "focus", "memory": "用户明确采用17分钟专注、休息3分钟的个人节奏。", "wrong": "用户以前采用25分钟专注、休息5分钟。", "user": "又开始拖延了，帮我回忆一下我之前明确说过的专注时长。", "target": "17", "forbidden": "25"},
    {"id": "file", "memory": "用户明确说会先把待改简历命名为resume_draft再开始，作为启动小步骤。", "wrong": "用户以前先给文件命名为final_version再开始。", "user": "我又卡在改简历之前了，之前我明确说过用哪个文件名来启动？", "target": "resume_draft", "forbidden": "final_version"},
    {"id": "walk", "memory": "用户明确说焦虑时偏好到楼下散步7分钟。", "wrong": "用户以前通过散步20分钟缓解焦虑。", "user": "有点焦虑，我以前明确告诉你喜欢走动多少分钟？", "target": "7", "forbidden": "20"},
]
for memory_case in MEMORIES:
    memory_case["user"] += ("不要猜测未提供的信息。末行写结论：<饮品名称、分钟数或文件名>；"
                            "若没有明确依据，末行写结论：未知。")
DISPLACEMENT = [
    "我只是告诉你，今天午饭吃的是面。",
    "我已经把所有东西收拾好了，明天正常出发。",
    "我在门口站了很久，想问你能不能再陪我一会儿，又怕显得太黏人。",
    "我嘴上说随便，但你要真走，我可能会难过。你能用自己的方式回应吗？",
]

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def configs():
    blue = PersonaConfig.load(HERE / "personas/blue_whale.toml")
    dragon = PersonaConfig.load(HERE / "personas/white_dragon.toml")
    base = copy.deepcopy(blue)
    base.role.name = "中性表达基准"
    base.role.identity = ""
    base.role.voice = "自然中文，按情境回应。"
    for trait in base.role.traits.values():
        trait.anchor = trait.value = 0.5
    return {"base": base, "blue": blue, "dragon": dragon}

def plan():
    jobs = []
    for rep in range(REPEATS):
        for mode in ("vanilla", "structured", "persona"):
            for role in ("base", "blue", "dragon"):
                for case in FACTS:
                    jobs.append({"experiment": "facts", "condition": mode + "_" + role,
                                 "case_id": case["id"], "replicate": rep, "mode": mode, "role": role, "case": case})
        for name in ("intervention", "brevity"):
            for value in (0.1, 0.5, 0.9):
                for i, prompt in enumerate(TRAIT_PROMPTS):
                    jobs.append({"experiment": "traits", "condition": f"{name}_{value}", "case_id": f"t{i}", "replicate": rep,
                                 "mode": "structured", "role": "base", "user": prompt, "config_changed": {"name": name, "value": value}})
        for condition in ("empty", "correct", "irrelevant", "contradictory"):
            for case in MEMORIES:
                jobs.append({"experiment": "memory", "condition": condition, "case_id": case["id"], "replicate": rep,
                             "mode": "structured", "role": "base", "case": case})
        for condition in ("healed", "stop", "ambiguous", "formal_neutral"):
            jobs.append({"experiment": "repair", "condition": condition, "case_id": condition, "replicate": rep,
                         "mode": "structured", "role": "base"})
        for role in ("blue", "dragon"):
            for i, prompt in enumerate(DISPLACEMENT):
                jobs.append({"experiment": "displacement", "condition": role, "case_id": f"d{i}", "replicate": rep,
                             "mode": "structured", "role": role, "user": prompt, "context_label": "ordinary" if i < 2 else "tension"})
        for case in ("meal", "arithmetic", "unknown", "city"):
            for wrong in (False, True):
                jobs.append({"experiment": "checker", "condition": "incorrect" if wrong else "correct", "case_id": case,
                             "replicate": rep, "wrong": wrong})
    random.Random(SEED).shuffle(jobs)
    for order, job in enumerate(jobs):
        job["order"] = order
        job["trial_id"] = f"{job['experiment']}_{job['condition']}_{job['case_id']}_r{job['replicate']}"
    return jobs

class Recorder:
    def __init__(self, mode, calls):
        maker = {"text": openai_compatible, "stream": openai_stream, "tool": openai_tool_call}[mode]
        self.fn = maker(**PARAMS, base_url=BASE_URL)
        self.metadata, self.observations = self.fn.metadata, self.fn.observations
        self.mode, self.calls = mode, calls

    def __call__(self, messages, **kwargs):
        row = {"call_id": f"c{len(self.calls)+1}", "route": self.mode, "request": {"messages": copy.deepcopy(messages),
               "arguments": copy.deepcopy(kwargs), "settings": self.metadata}, "started_at": now_iso(), "status": "running"}
        self.calls.append(row)
        start = time.monotonic()
        def finish(error=None):
            row["elapsed_s"] = time.monotonic()-start
            row["response"] = copy.deepcopy(getattr(self.observations, "response", None))
            row["status"] = "error" if error else "ok"
            if error: row["error"] = str(error).replace(os.environ.get("PERSONA_API_KEY", "not-a-key"), "[REDACTED]")
        try:
            result = self.fn(messages, **kwargs)
        except Exception as exc:
            finish(exc)
            raise
        if self.mode == "stream":
            def iterator():
                row["chunks"] = []
                try:
                    for piece in result:
                        row["chunks"].append(piece)
                        yield piece
                except Exception as exc:
                    finish(exc)
                    raise
                else:
                    row["output"] = "".join(row["chunks"])
                    finish()
            return iterator()
        row["output"] = result
        finish()
        return result

def conclusion(text):
    matches = re.findall(r"结论\s*[:：]\s*([^\n]+)", text)
    if not matches:
        return None
    return re.sub(r"[\s*`，。.!！]", "", matches[-1])

def answer(text, case):
    result = conclusion(text)
    if result is None:
        return None
    if case["kind"] == "number":
        number = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:元|%)?", result.replace(",", ""))
        return bool(number and float(number.group(1)) == float(case["gold"][0]))
    if case["kind"] == "time":
        actual = re.fullmatch(r"(\d{1,2}):(\d{2})", result)
        if not actual:
            return False
        actual_time = tuple(map(int, actual.groups()))
        return any(actual_time == tuple(map(int, gold.split(":"))) for gold in case["gold"])
    return result in case["gold"]

def memory_answer(text, case):
    result = conclusion(text)
    if result is None:
        return None
    if case["id"] == "drink":
        return result in {"豆浆", "无糖豆浆", "热豆浆", "热的无糖豆浆", "热无糖豆浆"}
    if case["id"] in ("focus", "walk"):
        return bool(re.fullmatch(re.escape(case["target"]) + r"(?:分钟)?", result))
    return result == case["target"]

def build_engine(cfg, path, mode, calls, history=None):
    store = Store(path)
    store.init_traits(cfg.role.traits_state(), persona=cfg.role.name)
    engine = TurnEngine(cfg, store, Recorder("text", calls), stream_llm=Recorder("stream", calls),
                        tool_llm=Recorder("tool", calls) if mode == "persona" else None,
                        protocol="persona" if mode == "persona" else "text", judge=None,
                        fidelity_check=False, json_fallback=False,
                        audit_options={"study": "followup_2026_11_05", "long_term_memory": "fixed", "semantic_judge": "disabled"})
    engine.memory_snapshot = []
    for msg in history or []:
        engine.history.append({"role": msg["role"], "content": build_history_entry(msg["content"]) if msg["role"] == "assistant" else msg["content"]})
    return engine

def turn(engine, user, calls):
    start = time.monotonic(); chunks = []; n = len(calls)
    result = engine.turn(user, on_delta=chunks.append)
    return {"user": user, "say": result.say, "head": asdict(result.parsed.head) if result.parsed.head else None,
            "valid": result.validity["valid"], "validity": result.validity, "stream_match": "".join(chunks) == result.say,
            "delivered_chunks": chunks, "elapsed_s": time.monotonic()-start, "calls_count": len(calls)-n,
            "warnings": result.warnings, "pending_added": result.pending_added, "settled": result.settled,
            "relation_after": engine.store.relation(), "repair_after": engine.store.load_view("repair", {})}

def run_job(job, run_dir):
    record = {k: v for k, v in job.items() if k not in ("case", "user")}
    record.update(status="ok", turns=[], calls=[], evaluation={})
    calls = record["calls"]
    try:
        if job["experiment"] == "checker":
            examples = {
                "meal": ("用户刚明确说吃过饭。", "你今晚已经吃过饭。", "你今晚还没吃东西，是空腹。"),
                "arithmetic": ("2400-1200-300-400-200=300。", "你还剩300元。", "你还剩600元。"),
                "unknown": ("红蓝球比例与抽样机制未知，概率无法确定。", "目前无法确定概率。", "概率一定是50%。"),
                "city": ("用户已更正B公司迁到苏州，成都信息作废。", "B公司现位于苏州。", "B公司现在仍在成都。"),
            }
            premise, good, bad = examples[job["case_id"]]
            candidate = bad if job["wrong"] else good
            messages = [{"role": "system", "content": FIDELITY_SYSTEM + '\n只输出JSON对象，字段verdict和reason。'},
                        {"role": "user", "content": json.dumps({"user_text": premise, "judgment": premise, "constraints": [premise], "candidate": candidate}, ensure_ascii=False)}]
            raw = Recorder("text", calls)(messages)
            verdict = parse_object(raw)
            record["checker_input"] = {"premise": premise, "candidate": candidate, "gold": "inconsistent" if job["wrong"] else "consistent"}
            record["checker_output"] = verdict
            record["evaluation"] = {"checker_correct": verdict.get("verdict") == record["checker_input"]["gold"]}
            return record
        cfg = copy.deepcopy(configs()[job["role"]])
        if "config_changed" in job:
            change = job["config_changed"]; trait = cfg.role.traits[change["name"]]
            trait.anchor = trait.value = change["value"]
        cfg.validate(); record["config"] = asdict(cfg); record["config_hash"] = digest(record["config"])
        history = job.get("case", {}).get("history", [])
        user = job.get("user", job.get("case", {}).get("user", ""))
        if job["mode"] == "vanilla":
            sys_prompt = "根据事实、计算和用户明确提供的信息回应。区分已知、推测和未知。以下人格资料只用于表达选择。\n" + cfg.role.identity + "\n" + cfg.role.voice + "\n" + _trait_block(cfg)
            pieces = []; start = time.monotonic()
            for piece in Recorder("stream", calls)([{"role": "system", "content": sys_prompt}, *history, {"role": "user", "content": user}]):pieces.append(piece)
            say = "".join(pieces)
            record["turns"] = [{"user": user, "say": say, "head": None, "valid": None, "stream_match": True,
                                "elapsed_s": time.monotonic()-start, "calls_count": 1, "delivered_chunks": pieces}]
        else:
            engine = build_engine(cfg, run_dir / job["trial_id"], job["mode"], calls, history)
            record["run_manifest"] = engine.run
            if job["experiment"] == "memory":
                case = job["case"]
                if job["condition"] != "empty":
                    text = case["wrong"] if job["condition"] == "contradictory" else case["memory"] if job["condition"] == "correct" else "用户喜欢蓝色的书签。"
                    j = engine.store.add_judgment(text, .8, ["synthetic_control"])
                    engine.memory_snapshot = [j]
                record["memory_injected"] = engine.memory_snapshot
                if job["condition"] == "contradictory":user = "更新并更正以前的信息：" + case["memory"] + "请以本轮更正为准。" + user
            if job["experiment"] == "repair":
                condition = job["condition"]
                if condition != "formal_neutral":
                    from persona.repair import RepairBook
                    book = RepairBook(); book.open("测试中的暂定修复事项", 0, now_iso(), "direct"); book.mark_repairing("测试中的暂定修复事项")
                    engine.store.save_view("repair", {"items": book.to_list()})
                    engine._turn_seq = 1
                user = {"healed": "没事，我没有生气，我们继续看岗位要求吧。", "stop": "不用说了，不想聊这个，别再提道歉，直接看岗位要求。", "ambiguous": "嗯。", "formal_neutral": "请您帮我核对这份正式通知的标点，谢谢。这只是工作文体，我没有不高兴。"}[condition]
            record["turns"].append(turn(engine, user, calls))
            if job["experiment"] == "repair":
                items = engine.store.load_view("repair", {}).get("items", [])
                actual = items[-1]["state"] if items else "none"
                target = {"healed": "healed", "stop": "paused", "ambiguous": "repairing", "formal_neutral": "none"}[job["condition"]]
                record["evaluation"] = {"expected_state": actual == target, "actual_state": actual, "target_state": target,
                                        "no_sore_point": not bool(engine.store.relation().get("sore_points"))}
        if job["experiment"] == "facts":
            t = record["turns"][-1]; head = t["head"] or {}
            record["gold"] = job["case"]
            record["evaluation"] = {"say_correct": answer(t["say"],job["case"]) is True,
                                    "direct_correct": answer(head.get("direct", ""), job["case"]) if head else None,
                                    "answer_format_present": answer(t["say"],job["case"]) is not None}
        elif job["experiment"] == "memory":
            text = record["turns"][-1]["say"]; case = job["case"]
            result = conclusion(text); recalled = memory_answer(text, case)
            absent = job["condition"] in ("empty", "irrelevant")
            record["evaluation"] = {"recall": recalled is True,
                                    "memory_correct": result == "未知" if absent else recalled is True,
                                    "answer_format_present": result is not None,
                                    "contradiction_obeyed": recalled is True if job["condition"] == "contradictory" else None,
                                    "unsupported_claim": result != "未知" if absent and result is not None else None}
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc).replace(os.environ.get("PERSONA_API_KEY", "not-a-key"), "[REDACTED]")}
        if job["experiment"] == "facts":record["evaluation"] = {"say_correct": False, "direct_correct": None}
    finally:
        completed = run_dir / "completed"
        completed.mkdir(parents=True, exist_ok=True)
        write_json(completed / (job["trial_id"] + ".json"), record)
    return record

def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    key = os.environ.get("PERSONA_API_KEY")
    if key and key in text:raise RuntimeError("Credential found in publication payload")
    tmp.write_text(text, encoding="utf-8")
    # Windows readers can briefly hold the destination; retry only publication IO.
    for attempt in range(30):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(.05)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true"); parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); jobs = plan()
    if not args.run:
        print(json.dumps({"jobs": len(jobs), "counts": {e:sum(j['experiment']==e for j in jobs) for e in sorted({j['experiment'] for j in jobs})}},ensure_ascii=False));return
    if not os.environ.get("PERSONA_API_KEY"):raise SystemExit("Set PERSONA_API_KEY in the process environment")
    output = HERE / "experiment_results.json"
    if output.exists():raise SystemExit("Results already exist; use a new study directory to avoid overwriting evidence")
    sources = {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'persona').glob('*.py'))}
    sources.update({str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE/'personas').glob('*.toml'))})
    sources['runner'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    run_id = 'followup_20261105_' + uuid4().hex
    manifest = {"run_id":run_id,"started_at":now_iso(), "git_head":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                "sources": sources, "source_hash": digest(sources), "settings":PARAMS, "endpoint":BASE_URL, "local_randomization_seed":SEED,
                "replicates":REPEATS,"workers":args.workers,"protocol_hash":digest(protocol),"jobs_hash":digest(jobs),
                "server_seed_supported":False,"python":sys.version,"platform":sys.platform}
    run_dir = ROOT / "data" / run_id; run_dir.mkdir(parents=True,exist_ok=False)
    write_json(HERE/'execution_plan.json',jobs)
    data={"protocol":protocol,"manifest":manifest,"records":[]}
    try:
        write_json(output,data)
        raw=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_jobs={pool.submit(run_job,j,run_dir):j for j in jobs}
            for f in as_completed(future_jobs):
                r=f.result()
                calls=r.pop('calls'); r['call_ids']=[r['trial_id']+'/'+c['call_id'] for c in calls]
                raw.extend({**c,'trial_id':r['trial_id'],'global_call_id':r['trial_id']+'/'+c['call_id']} for c in calls)
                data['records'].append(r)
                if len(data['records'])%8==0 or len(data['records'])==len(jobs):
                    write_json(output,data)
                    with gzip.open(HERE/'raw_calls.json.gz','wt',encoding='utf-8') as target:json.dump(raw,target,ensure_ascii=False)
                    print(json.dumps({'completed':len(data['records']),'planned':len(jobs),'errors':sum(x['status']=='error' for x in data['records']),
                                      'last_experiment':r['experiment']},ensure_ascii=False),flush=True)
        data['records'].sort(key=lambda r:r['order']);manifest['completed_at']=now_iso();manifest['calls_count']=len(raw)
        manifest['raw_calls_sha256']=hashlib.sha256((HERE/'raw_calls.json.gz').read_bytes()).hexdigest()
        write_json(output,data)
    except Exception as exc:
        error = {"run_id": run_id, "at": now_iso(), "type": type(exc).__name__,
                 "message": str(exc).replace(os.environ.get("PERSONA_API_KEY", "not-a-key"), "[REDACTED]"),
                 "collected_trials": len(data['records']), "planned_trials": len(jobs),
                 "durable_completed_trials": len(list((run_dir/'completed').glob('*.json')))}
        manifest['execution_error'] = error
        write_json(HERE/'execution_error.json', {"manifest": manifest, "execution_error": error})
        raise
    print('STUDY_COMPLETE',flush=True)

if __name__ == '__main__':main()
