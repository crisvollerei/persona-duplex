"""Application-declared constraints compiled into auditable factual references.

Only a finite all-of rule and simultaneous resource budget are evaluated here.
This compiler does not infer user intent, validate source truth or judge SAY.
"""
from dataclasses import asdict, dataclass
import json
import re


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[^\W\d][\w.-]{0,63}', value):
        raise ValueError('constraint names must be short identifiers')


def _source(value):
    if not isinstance(value, str):
        raise TypeError('source must be text supplied by the application')


def _items(value, kind, label):
    if not isinstance(value, (tuple, list)) or any(not isinstance(v, kind) for v in value):
        raise TypeError(label+' must contain '+kind.__name__)
    return tuple(value)


@dataclass(frozen=True)
class BoolFact:
    name: str
    value: bool | None
    source: str = ''

    def __post_init__(self):
        _name(self.name)
        if type(self.value) not in (bool, type(None)):
            raise TypeError('fact value must be bool or None for unknown')
        _source(self.source)


@dataclass(frozen=True)
class AllOfRule:
    name: str
    requires: tuple[str, ...]
    source: str = ''

    def __post_init__(self):
        _name(self.name)
        if not isinstance(self.requires, (tuple, list)) or not self.requires:
            raise ValueError('an all-of rule needs explicit prerequisites')
        for name in self.requires:
            _name(name)
        if len(set(self.requires)) != len(self.requires):
            raise ValueError('duplicate prerequisite')
        object.__setattr__(self, 'requires', tuple(self.requires))
        _source(self.source)


@dataclass(frozen=True)
class ResourceBudget:
    name: str
    available: int
    units: int
    per_unit: int = 1
    source: str = ''

    def __post_init__(self):
        _name(self.name)
        for name in ('available', 'units', 'per_unit'):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == 'per_unit' else 0):
                raise ValueError(name+' must be an integer within its nonnegative/positive domain')
        _source(self.source)


@dataclass(frozen=True)
class ConstraintContext:
    facts: tuple[BoolFact, ...] = ()
    rules: tuple[AllOfRule, ...] = ()
    resources: tuple[ResourceBudget, ...] = ()

    def __post_init__(self):
        for field, kind in (('facts', BoolFact), ('rules', AllOfRule), ('resources', ResourceBudget)):
            values = _items(getattr(self, field), kind, field)
            if len({v.name for v in values}) != len(values):
                raise ValueError('duplicate '+field+' name')
            object.__setattr__(self, field, values)
        names = {f.name for f in self.facts}
        if any(name not in names for rule in self.rules for name in rule.requires):
            raise ValueError('rule refers to an undeclared fact; declare unknown explicitly')
        if not self.rules and not self.resources:
            raise ValueError('at least one rule or resource budget is required')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {'facts', 'rules', 'resources'}:
            raise ValueError('expected an explicit constraint context object')
        parsed = {}
        for field, kind in (('facts', BoolFact), ('rules', AllOfRule), ('resources', ResourceBudget)):
            rows = value.get(field, [])
            if not isinstance(rows, (list, tuple)) or any(not isinstance(row, dict) for row in rows):
                raise TypeError(field+' must be a sequence of objects')
            parsed[field] = tuple(kind(**row) for row in rows)
        return cls(**parsed)

    def reference(self):
        indexed = {f.name: f for f in self.facts}
        rules = []
        for rule in self.rules:
            values = [indexed[name].value for name in rule.requires]
            truth = 'false' if False in values else 'unknown' if None in values else 'true'
            rules.append(dict(name=rule.name, source=rule.source, requires=list(rule.requires),
                truth=truth, unknown=[n for n in rule.requires if indexed[n].value is None],
                false=[n for n in rule.requires if indexed[n].value is False],
                context_not_prerequisites=[f.name for f in self.facts if f.name not in rule.requires]))
        resources = [dict(name=r.name, source=r.source, available=r.available, units=r.units,
            per_unit=r.per_unit, required=r.units*r.per_unit, capacity=r.available//r.per_unit,
            shortfall=max(0, r.units*r.per_unit-r.available),
            unsupported_units=max(0, r.units-r.available//r.per_unit)) for r in self.resources]
        inputs=dict(facts=[asdict(f) for f in self.facts],
            rules=[dict(name=r.name, requires=list(r.requires), source=r.source) for r in self.rules],
            resources=[asdict(r) for r in self.resources])
        return dict(version=1, inputs=inputs, rules=rules, resources=resources,
            scope='application_declared_closed_rules_not_verified_source_truth',
            output_validation='not_assessed')

    def prompt(self):
        return ('[应用声明的条件与资源参考 · 本轮有效]\n'
            '以下是给定输入的程序推导，不是已执行事务，也不是对回复正确性的认证。'
            'all-of规则当且仅当列出的前提都真才为真；有已知假则为假，否则有未知则保持未知。'
            '背景事实不自动成为新必要条件，未知不等于假或已满足。'
            '资源按同时占用计算，capacity是最多可支持单位数；不足部分不能凭空借入或假定复用。'
            '依据这些参考完成当前请求，条件化方案需明确假设；人格仍决定表达取舍。\n'
            +json.dumps(self.reference(), ensure_ascii=False, sort_keys=True, allow_nan=False))
