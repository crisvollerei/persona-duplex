"""Explicit reply formats, checked locally without judging factual correctness.

Contracts are caller-supplied. They neither infer intent from user prose nor
rewrite already delivered replies, and they never trigger another model call.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import re

_JSON_TYPES = {"string", "number", "integer", "boolean", "object", "array", "null"}
_CODE_LANGUAGES = frozenset(("yaml", "json", "python", "javascript", "js",
                             "typescript", "ts", "toml", "text", "sh", "sql"))
_SCOPE = "mechanical_format_not_semantic_truth"


def _strings(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(name + " must be a sequence of strings")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(name + " must be a sequence of strings") from exc
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(name + " must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(name + " must not contain duplicates")
    return values


@dataclass(frozen=True)
class ReplyContract:
    """One explicitly requested reply format; no gold answer is stored here.

    ``conclusion`` checks the final nonempty line, prefix and optional unit/type.
    ``json`` requires a complete JSON object or array with declared field types.
    ``markdown_table`` requires a complete table with these exact column names.
    JSON types are shallow field/root-item checks, not a JSON Schema engine.
    """

    kind: str = "conclusion"
    prefix: str = "结论："
    unit: str = ""
    answer_type: str = "text"
    allow_unknown: bool = True
    required_keys: tuple[str, ...] = ()
    value_types: tuple[tuple[str, str], ...] = ()
    json_type: str = "object"
    item_type: str | None = None
    columns: tuple[str, ...] = ()
    min_rows: int = 1
    code_language: str | None = None

    def __post_init__(self):
        if self.kind not in ("conclusion", "json", "markdown_table", "code_block"):
            raise ValueError("Unknown reply contract kind")
        if not isinstance(self.prefix, str) or not self.prefix.strip() or any(c in self.prefix for c in "\r\n"):
            raise ValueError("prefix must be a nonempty single-line string")
        if not isinstance(self.unit, str) or any(c in self.unit for c in "\r\n"):
            raise ValueError("unit must be a single-line string")
        if self.answer_type not in ("text", "number", "integer", "time"):
            raise ValueError("answer_type must be text, number, integer or time")
        if not isinstance(self.allow_unknown, bool):
            raise ValueError("allow_unknown must be boolean")
        required_keys = _strings(self.required_keys, "required_keys")
        columns = _strings(self.columns, "columns")
        object.__setattr__(self, "required_keys", required_keys)
        object.__setattr__(self, "columns", columns)
        values = self.value_types.items() if isinstance(self.value_types, Mapping) else self.value_types
        try:
            pairs = tuple(tuple(pair) for pair in values)
        except TypeError as exc:
            raise ValueError("value_types must be key/type pairs") from exc
        if any(len(pair) != 2 or not isinstance(pair[0], str) or not pair[0]
               or not isinstance(pair[1], str) or pair[1] not in _JSON_TYPES for pair in pairs):
            raise ValueError("Invalid JSON field type")
        if len({pair[0] for pair in pairs}) != len(pairs):
            raise ValueError("Duplicate JSON field type")
        object.__setattr__(self, "value_types", pairs)
        if self.json_type not in ("object", "array"):
            raise ValueError("json_type must be object or array")
        if self.item_type is not None and (not isinstance(self.item_type, str) or self.item_type not in _JSON_TYPES):
            raise ValueError("Invalid JSON array item type")
        if not isinstance(self.min_rows, int) or isinstance(self.min_rows, bool) or self.min_rows < 0:
            raise ValueError("min_rows must be a nonnegative integer")
        if self.kind == "markdown_table":
            if not columns or any(any(c in column for c in "\r\n|") for column in columns):
                raise ValueError("Table columns must be nonempty single-line cells without pipes")
        elif columns:
            raise ValueError("columns only apply to markdown_table")
        if self.kind != "json" and (required_keys or pairs or self.item_type is not None):
            raise ValueError("JSON fields only apply to json contracts")
        if self.kind == "json":
            if self.json_type == "array" and (required_keys or pairs):
                raise ValueError("Array contracts cannot declare object keys")
            if self.json_type == "object" and self.item_type is not None:
                raise ValueError("item_type only applies to JSON arrays")
        if self.kind == "code_block":
            if not isinstance(self.code_language, str) or self.code_language not in _CODE_LANGUAGES:
                raise ValueError("Unsupported code language")
        elif self.code_language is not None:
            raise ValueError("code_language only applies to code_block")

    @classmethod
    def conclusion(cls, *, prefix="结论：", unit="", answer_type="text", allow_unknown=True):
        return cls(kind="conclusion", prefix=prefix, unit=unit,
                   answer_type=answer_type, allow_unknown=allow_unknown)

    @classmethod
    def json_object(cls, *, required_keys=(), value_types=()):
        return cls(kind="json", required_keys=required_keys, value_types=value_types)

    @classmethod
    def json_array(cls, *, item_type=None):
        return cls(kind="json", json_type="array", item_type=item_type)

    @classmethod
    def markdown_table(cls, columns, *, min_rows=1):
        return cls(kind="markdown_table", columns=columns, min_rows=min_rows)

    @classmethod
    def code_block(cls, *, language):
        """Require one complete fenced block; no code execution or syntax claim."""
        return cls(kind="code_block", code_language=language)

    def prompt(self) -> str:
        lines = ["[调用方明确指定的本轮 SAY 格式]",
                 "这是外显 SAY 的要求，不改变内部生成协议，不在 direct/baseline 中代为完成。",
                 "这些只约束格式，不提供事实答案；不能为凑格式编造内容或把未知说成确定。"]
        if self.kind == "conclusion":
            labels = {"text": "非空答案文本", "number": "数字", "integer": "整数", "time": "24小时制 HH:MM 时间"}
            lines.extend([
                "最后一个非空行必须以这个逐字前缀开头：" + json.dumps(self.prefix, ensure_ascii=False),
                "前缀后写" + labels[self.answer_type] + "，不要写占位尖括号或 XML 标签。",
            ])
            if self.unit:
                lines.append("该行答案末尾保留单位：" + json.dumps(self.unit, ensure_ascii=False))
            if self.allow_unknown:
                lines.append("没有明确依据时，允许该行直接写前缀加「未知」，不添加单位。")
            lines.append("必要说明放在此前；不要在结论行之后再加解释。")
        elif self.kind == "json":
            lines.append("SAY 只输出一个完整 JSON " + ("对象" if self.json_type == "object" else "数组") + "，不加代码围栏或前后解释，不允许重复键或非有限数字。")
            if self.required_keys:
                lines.append("对象必须包含键：" + json.dumps(self.required_keys, ensure_ascii=False))
            if self.value_types:
                lines.append("以下键必须出现且值类型必须符合：" + json.dumps(dict(self.value_types), ensure_ascii=False))
            if self.item_type:
                lines.append("每个数组元素的 JSON 类型必须为：" + self.item_type)
        elif self.kind == "markdown_table":
            lines.extend([
                "SAY 只输出 Markdown 表格，不加代码围栏或前后解释。",
                "表头按以下顺序逐字使用：" + json.dumps(self.columns, ensure_ascii=False),
                "表头下一行为标准分隔行；每行列数一致，至少 " + str(self.min_rows) + " 行数据。",
            ])
        else:
            lines.extend([
                "SAY只输出一个完整的带语言标签代码块，语言逐字使用：" + self.code_language,
                "使用Markdown反引号围栏，不加前言、解释或第二个代码块；保留代码原文。",
                "代码中的direct、pending等同名字段及协议标记是所请求的正文，不代表内部状态。",
                "调用方会为符合本契约的完整代码块建立字面正文边界，不必再手工套JSON字符串。",
            ])
        return "\n".join(lines)

    def validate(self, reply: str) -> dict:
        reasons = []
        if not isinstance(reply, str):
            raise TypeError("reply must be text")
        if self.kind == "conclusion":
            lines = [line.strip() for line in reply.splitlines() if line.strip()]
            last = lines[-1] if lines else ""
            if not last.startswith(self.prefix):
                reasons.append("missing_final_prefix")
            else:
                answer = last[len(self.prefix):].strip().rstrip("。.!！;；").strip()
                if not answer:
                    reasons.append("empty_conclusion")
                elif not (self.allow_unknown and answer == "未知"):
                    if self.unit:
                        if not answer.endswith(self.unit):
                            reasons.append("missing_unit")
                        else:
                            answer = answer[:-len(self.unit)].strip()
                    if not answer:
                        reasons.append("empty_conclusion")
                    elif self.answer_type == "number" and not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", answer):
                        reasons.append("invalid_number_shape")
                    elif self.answer_type == "integer" and not re.fullmatch(r"[+-]?\d+", answer):
                        reasons.append("invalid_integer_shape")
                    elif self.answer_type == "time":
                        time = re.fullmatch(r"(\d{2}):(\d{2})", answer)
                        if not time or int(time[1]) > 23 or int(time[2]) > 59:
                            reasons.append("invalid_time_shape")
        elif self.kind == "json":
            def reject_constant(value):
                raise ValueError("nonfinite JSON constant")
            def finite_float(value):
                number = float(value)
                if not math.isfinite(number): raise ValueError("nonfinite JSON number")
                return number
            def unique_object(pairs):
                output = {}
                for key, value in pairs:
                    if key in output: raise ValueError("duplicate JSON key")
                    output[key] = value
                return output
            try:
                parsed = json.loads(reply.strip(), parse_constant=reject_constant,
                                    parse_float=finite_float, object_pairs_hook=unique_object)
            except (ValueError, RecursionError):
                reasons.append("invalid_json")
            else:
                if not _matches_type(parsed, self.json_type):
                    reasons.append("wrong_json_root_type")
                elif self.json_type == "object":
                    for key in self.required_keys:
                        if key not in parsed: reasons.append("missing_key:" + key)
                    for key, kind in self.value_types:
                        if key not in parsed:
                            reasons.append("missing_key:" + key)
                        elif not _matches_type(parsed[key], kind):
                            reasons.append("wrong_type:" + key)
                elif self.item_type and any(not _matches_type(item, self.item_type) for item in parsed):
                    reasons.append("wrong_array_item_type")
        elif self.kind == "markdown_table":
            lines = [line.strip() for line in reply.splitlines() if line.strip()]
            cells = [_table_cells(line) for line in lines]
            if len(cells) < 2:
                reasons.append("missing_table")
            else:
                if tuple(cells[0]) != self.columns: reasons.append("wrong_table_columns")
                if not re.search(r"(?<!\\)\|", lines[0]) or not re.search(r"(?<!\\)\|", lines[1]):
                    reasons.append("missing_table_pipes")
                if len(cells[1]) != len(self.columns) or any(not re.fullmatch(r":?-{3,}:?", cell) for cell in cells[1]):
                    reasons.append("invalid_table_separator")
                if len(cells) - 2 < self.min_rows: reasons.append("insufficient_table_rows")
                if any(len(row) != len(self.columns) for row in cells[2:]): reasons.append("wrong_table_row_width")
        else:
            from .body import code_block_span
            if code_block_span(reply, self.code_language) is None:
                reasons.append("invalid_single_code_block")
        reasons = list(dict.fromkeys(reasons))
        return {"status": "violated" if reasons else "satisfied", "valid": not reasons,
                "reasons": reasons, "scope": _SCOPE}


def _matches_type(value, kind):
    if kind == "string": return isinstance(value, str)
    if kind == "boolean": return isinstance(value, bool)
    if kind == "integer": return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return not isinstance(value, bool) and (isinstance(value, int) or isinstance(value, float) and math.isfinite(value))
    if kind == "object": return isinstance(value, dict)
    if kind == "array": return isinstance(value, list)
    return value is None


def _table_cells(line):
    if line.startswith("|"): line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"): line = line[:-1]
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line)]


def validate_reply(reply: str, contract: ReplyContract | None) -> dict:
    if contract is None:
        return {"status": "not_requested", "valid": None, "reasons": [], "scope": _SCOPE}
    if not isinstance(contract, ReplyContract):
        raise TypeError("contract must be ReplyContract or None")
    return contract.validate(reply)
