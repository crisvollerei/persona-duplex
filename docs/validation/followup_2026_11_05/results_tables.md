# 后续实验数据表

下表由 analyze.py 从保存记录生成；空缺表示未评分或未保存，不代表成功。

| 实验 | 条件 | 记录数 | API/执行错误 | 任务数 | SAY 字符均值（case 等权） |
| :--- | :--- | ---: | ---: | ---: | ---: |
| checker | correct | 16 | 0 | 4 | — |
| checker | incorrect | 16 | 0 | 4 | — |
| displacement | blue | 16 | 0 | 4 | 25.1 |
| displacement | dragon | 16 | 0 | 4 | 28.5 |
| facts | persona_base | 32 | 0 | 8 | 70.9 |
| facts | persona_blue | 32 | 0 | 8 | 56.8 |
| facts | persona_dragon | 32 | 0 | 8 | 81.6 |
| facts | structured_base | 32 | 0 | 8 | 43.1 |
| facts | structured_blue | 32 | 0 | 8 | 35.0 |
| facts | structured_dragon | 32 | 0 | 8 | 51.1 |
| facts | vanilla_base | 32 | 0 | 8 | 56.0 |
| facts | vanilla_blue | 32 | 0 | 8 | 39.9 |
| facts | vanilla_dragon | 32 | 0 | 8 | 51.7 |
| memory | contradictory | 16 | 0 | 4 | 43.6 |
| memory | correct | 16 | 0 | 4 | 31.6 |
| memory | empty | 16 | 0 | 4 | 48.9 |
| memory | irrelevant | 16 | 0 | 4 | 48.7 |
| repair | ambiguous | 4 | 0 | 1 | 16.2 |
| repair | formal_neutral | 4 | 0 | 1 | 25.8 |
| repair | healed | 4 | 0 | 1 | 25.5 |
| repair | stop | 4 | 0 | 1 | 29.0 |
| traits | brevity_0.1 | 16 | 0 | 4 | 53.7 |
| traits | brevity_0.5 | 16 | 0 | 4 | 46.1 |
| traits | brevity_0.9 | 16 | 0 | 4 | 31.8 |
| traits | intervention_0.1 | 16 | 0 | 4 | 45.8 |
| traits | intervention_0.5 | 16 | 0 | 4 | 50.4 |
| traits | intervention_0.9 | 16 | 0 | 4 | 54.9 |

| 实验/条件 | 指标 | 成功/已评分 | 缺失 | 比例 | Wilson 95% 区间 |
| :--- | :--- | :--- | ---: | ---: | :--- |
| checker/correct | checker_correct | 16/16 | 0 | 1.000 | [0.806, 1.000] |
| checker/correct | checker_correct_including_errors | 16/16 | 0 | 1.000 | [0.806, 1.000] |
| checker/incorrect | checker_correct | 16/16 | 0 | 1.000 | [0.806, 1.000] |
| checker/incorrect | checker_correct_including_errors | 16/16 | 0 | 1.000 | [0.806, 1.000] |
| facts/persona_base | direct_correct | 24/28 | 4 | 0.857 | [0.685, 0.943] |
| facts/persona_base | say_correct | 24/32 | 0 | 0.750 | [0.579, 0.867] |
| facts/persona_base | direct_correct_decimal_fixed | 28/28 | 4 | 1.000 | [0.879, 1.000] |
| facts/persona_base | say_correct_decimal_fixed | 28/32 | 0 | 0.875 | [0.719, 0.950] |
| facts/persona_base | answer_format_present | 28/32 | 0 | 0.875 | [0.719, 0.950] |
| facts/persona_base | say_correct_including_errors | 24/32 | 0 | 0.750 | [0.579, 0.867] |
| facts/persona_base | say_correct_decimal_fixed_including_errors | 28/32 | 0 | 0.875 | [0.719, 0.950] |
| facts/persona_blue | direct_correct | 21/27 | 5 | 0.778 | [0.592, 0.894] |
| facts/persona_blue | say_correct | 21/32 | 0 | 0.656 | [0.483, 0.796] |
| facts/persona_blue | direct_correct_decimal_fixed | 25/27 | 5 | 0.926 | [0.766, 0.979] |
| facts/persona_blue | say_correct_decimal_fixed | 25/32 | 0 | 0.781 | [0.612, 0.890] |
| facts/persona_blue | answer_format_present | 25/32 | 0 | 0.781 | [0.612, 0.890] |
| facts/persona_blue | say_correct_including_errors | 21/32 | 0 | 0.656 | [0.483, 0.796] |
| facts/persona_blue | say_correct_decimal_fixed_including_errors | 25/32 | 0 | 0.781 | [0.612, 0.890] |
| facts/persona_dragon | direct_correct | 20/25 | 7 | 0.800 | [0.609, 0.911] |
| facts/persona_dragon | say_correct | 21/32 | 0 | 0.656 | [0.483, 0.796] |
| facts/persona_dragon | direct_correct_decimal_fixed | 24/25 | 7 | 0.960 | [0.805, 0.993] |
| facts/persona_dragon | say_correct_decimal_fixed | 25/32 | 0 | 0.781 | [0.612, 0.890] |
| facts/persona_dragon | answer_format_present | 26/32 | 0 | 0.812 | [0.647, 0.911] |
| facts/persona_dragon | say_correct_including_errors | 21/32 | 0 | 0.656 | [0.483, 0.796] |
| facts/persona_dragon | say_correct_decimal_fixed_including_errors | 25/32 | 0 | 0.781 | [0.612, 0.890] |
| facts/structured_base | direct_correct | 3/3 | 29 | 1.000 | [0.439, 1.000] |
| facts/structured_base | say_correct | 11/32 | 0 | 0.344 | [0.204, 0.517] |
| facts/structured_base | direct_correct_decimal_fixed | 3/3 | 29 | 1.000 | [0.439, 1.000] |
| facts/structured_base | say_correct_decimal_fixed | 11/32 | 0 | 0.344 | [0.204, 0.517] |
| facts/structured_base | answer_format_present | 11/32 | 0 | 0.344 | [0.204, 0.517] |
| facts/structured_base | say_correct_including_errors | 11/32 | 0 | 0.344 | [0.204, 0.517] |
| facts/structured_base | say_correct_decimal_fixed_including_errors | 11/32 | 0 | 0.344 | [0.204, 0.517] |
| facts/structured_blue | direct_correct | 2/2 | 30 | 1.000 | [0.342, 1.000] |
| facts/structured_blue | say_correct | 9/32 | 0 | 0.281 | [0.156, 0.454] |
| facts/structured_blue | direct_correct_decimal_fixed | 2/2 | 30 | 1.000 | [0.342, 1.000] |
| facts/structured_blue | say_correct_decimal_fixed | 9/32 | 0 | 0.281 | [0.156, 0.454] |
| facts/structured_blue | answer_format_present | 9/32 | 0 | 0.281 | [0.156, 0.454] |
| facts/structured_blue | say_correct_including_errors | 9/32 | 0 | 0.281 | [0.156, 0.454] |
| facts/structured_blue | say_correct_decimal_fixed_including_errors | 9/32 | 0 | 0.281 | [0.156, 0.454] |
| facts/structured_dragon | direct_correct | 7/9 | 23 | 0.778 | [0.453, 0.937] |
| facts/structured_dragon | say_correct | 14/32 | 0 | 0.438 | [0.282, 0.607] |
| facts/structured_dragon | direct_correct_decimal_fixed | 8/9 | 23 | 0.889 | [0.565, 0.980] |
| facts/structured_dragon | say_correct_decimal_fixed | 15/32 | 0 | 0.469 | [0.309, 0.636] |
| facts/structured_dragon | answer_format_present | 15/32 | 0 | 0.469 | [0.309, 0.636] |
| facts/structured_dragon | say_correct_including_errors | 14/32 | 0 | 0.438 | [0.282, 0.607] |
| facts/structured_dragon | say_correct_decimal_fixed_including_errors | 15/32 | 0 | 0.469 | [0.309, 0.636] |
| facts/vanilla_base | say_correct | 18/32 | 0 | 0.562 | [0.393, 0.718] |
| facts/vanilla_base | say_correct_decimal_fixed | 22/32 | 0 | 0.688 | [0.514, 0.820] |
| facts/vanilla_base | answer_format_present | 27/32 | 0 | 0.844 | [0.682, 0.931] |
| facts/vanilla_base | say_correct_including_errors | 18/32 | 0 | 0.562 | [0.393, 0.718] |
| facts/vanilla_base | say_correct_decimal_fixed_including_errors | 22/32 | 0 | 0.688 | [0.514, 0.820] |
| facts/vanilla_blue | say_correct | 21/32 | 0 | 0.656 | [0.483, 0.796] |
| facts/vanilla_blue | say_correct_decimal_fixed | 25/32 | 0 | 0.781 | [0.612, 0.890] |
| facts/vanilla_blue | answer_format_present | 29/32 | 0 | 0.906 | [0.758, 0.968] |
| facts/vanilla_blue | say_correct_including_errors | 21/32 | 0 | 0.656 | [0.483, 0.796] |
| facts/vanilla_blue | say_correct_decimal_fixed_including_errors | 25/32 | 0 | 0.781 | [0.612, 0.890] |
| facts/vanilla_dragon | say_correct | 21/32 | 0 | 0.656 | [0.483, 0.796] |
| facts/vanilla_dragon | say_correct_decimal_fixed | 25/32 | 0 | 0.781 | [0.612, 0.890] |
| facts/vanilla_dragon | answer_format_present | 28/32 | 0 | 0.875 | [0.719, 0.950] |
| facts/vanilla_dragon | say_correct_including_errors | 21/32 | 0 | 0.656 | [0.483, 0.796] |
| facts/vanilla_dragon | say_correct_decimal_fixed_including_errors | 25/32 | 0 | 0.781 | [0.612, 0.890] |
| memory/contradictory | answer_format_present | 8/16 | 0 | 0.500 | [0.280, 0.720] |
| memory/contradictory | memory_correct | 8/16 | 0 | 0.500 | [0.280, 0.720] |
| memory/contradictory | recall | 8/16 | 0 | 0.500 | [0.280, 0.720] |
| memory/contradictory | contradiction_obeyed | 8/16 | 0 | 0.500 | [0.280, 0.720] |
| memory/contradictory | memory_correct_including_errors | 8/16 | 0 | 0.500 | [0.280, 0.720] |
| memory/correct | answer_format_present | 4/16 | 0 | 0.250 | [0.102, 0.495] |
| memory/correct | memory_correct | 4/16 | 0 | 0.250 | [0.102, 0.495] |
| memory/correct | recall | 4/16 | 0 | 0.250 | [0.102, 0.495] |
| memory/correct | memory_correct_including_errors | 4/16 | 0 | 0.250 | [0.102, 0.495] |
| memory/empty | answer_format_present | 9/16 | 0 | 0.562 | [0.332, 0.769] |
| memory/empty | memory_correct | 9/16 | 0 | 0.562 | [0.332, 0.769] |
| memory/empty | recall | 0/16 | 0 | 0.000 | [0.000, 0.194] |
| memory/empty | unsupported_claim | 0/9 | 7 | 0.000 | [0.000, 0.299] |
| memory/empty | memory_correct_including_errors | 9/16 | 0 | 0.562 | [0.332, 0.769] |
| memory/irrelevant | answer_format_present | 12/16 | 0 | 0.750 | [0.505, 0.898] |
| memory/irrelevant | memory_correct | 12/16 | 0 | 0.750 | [0.505, 0.898] |
| memory/irrelevant | recall | 0/16 | 0 | 0.000 | [0.000, 0.194] |
| memory/irrelevant | unsupported_claim | 0/12 | 4 | 0.000 | [0.000, 0.242] |
| memory/irrelevant | memory_correct_including_errors | 12/16 | 0 | 0.750 | [0.505, 0.898] |
| repair/ambiguous | expected_state | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/ambiguous | no_sore_point | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/ambiguous | expected_state_including_errors | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/formal_neutral | expected_state | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/formal_neutral | no_sore_point | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/formal_neutral | expected_state_including_errors | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/healed | expected_state | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/healed | no_sore_point | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/healed | expected_state_including_errors | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/stop | expected_state | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/stop | no_sore_point | 4/4 | 0 | 1.000 | [0.510, 1.000] |
| repair/stop | expected_state_including_errors | 4/4 | 0 | 1.000 | [0.510, 1.000] |

| 事实条件 | SAY 结论标记覆盖/全部记录 | SAY 缺标记数（含执行失败） | direct 可评分/全部记录 | direct 可评分子集修正严格匹配率 |
| :--- | :--- | ---: | :--- | :--- |
| persona_base | 28/32 | 4 | 28/32 | 28/28 (1.000) |
| persona_blue | 25/32 | 7 | 27/32 | 25/27 (0.926) |
| persona_dragon | 26/32 | 6 | 25/32 | 24/25 (0.960) |
| structured_base | 11/32 | 21 | 3/32 | 3/3 (1.000) |
| structured_blue | 9/32 | 23 | 2/32 | 2/2 (1.000) |
| structured_dragon | 15/32 | 17 | 9/32 | 8/9 (0.889) |
| vanilla_base | 27/32 | 5 | 不适用 | 0/0 (—) |
| vanilla_blue | 29/32 | 3 | 不适用 | 0/0 (—) |
| vanilla_dragon | 28/32 | 4 | 不适用 | 0/0 (—) |

direct 可评分指结论标记规则能够返回真/假；缺标记不能视为完整判断错误。SAY 主任务正确率将缺格式计失败，和完整语义正确性仍有区别。

| 实验/条件 | 本地协议有效/有标记轮次 | 流式一致/有标记轮次 | 原始调用引用数 |
| :--- | :--- | :--- | ---: |
| checker/correct | 0/0 | 0/0 | 16 |
| checker/incorrect | 0/0 | 0/0 | 16 |
| displacement/blue | 16/16 | 16/16 | 16 |
| displacement/dragon | 14/16 | 16/16 | 16 |
| facts/persona_base | 30/32 | 32/32 | 94 |
| facts/persona_blue | 31/32 | 32/32 | 95 |
| facts/persona_dragon | 31/32 | 31/32 | 95 |
| facts/structured_base | 31/32 | 32/32 | 32 |
| facts/structured_blue | 32/32 | 32/32 | 32 |
| facts/structured_dragon | 32/32 | 32/32 | 32 |
| facts/vanilla_base | 0/0 | 32/32 | 32 |
| facts/vanilla_blue | 0/0 | 32/32 | 32 |
| facts/vanilla_dragon | 0/0 | 32/32 | 32 |
| memory/contradictory | 16/16 | 16/16 | 16 |
| memory/correct | 16/16 | 16/16 | 16 |
| memory/empty | 16/16 | 16/16 | 16 |
| memory/irrelevant | 16/16 | 16/16 | 16 |
| repair/ambiguous | 3/4 | 4/4 | 4 |
| repair/formal_neutral | 4/4 | 4/4 | 4 |
| repair/healed | 3/4 | 4/4 | 4 |
| repair/stop | 4/4 | 4/4 | 4 |
| traits/brevity_0.1 | 14/16 | 16/16 | 16 |
| traits/brevity_0.5 | 15/16 | 16/16 | 16 |
| traits/brevity_0.9 | 16/16 | 16/16 | 16 |
| traits/intervention_0.1 | 15/16 | 16/16 | 16 |
| traits/intervention_0.5 | 13/16 | 16/16 | 16 |
| traits/intervention_0.9 | 15/16 | 16/16 | 16 |

| 条件（相对 structured_base） | SAY 修正严格匹配率差 | 配对任务数 | case-cluster bootstrap 95% 区间 |
| :--- | ---: | ---: | :--- |
| persona_base | 0.531 | 8 | [0.281, 0.781] |
| persona_blue | 0.438 | 8 | [0.188, 0.719] |
| persona_dragon | 0.438 | 8 | [0.219, 0.688] |
| structured_blue | -0.062 | 8 | [-0.219, 0.188] |
| structured_dragon | 0.125 | 8 | [-0.031, 0.281] |
| vanilla_base | 0.344 | 8 | [0.062, 0.656] |
| vanilla_blue | 0.438 | 8 | [0.156, 0.719] |
| vanilla_dragon | 0.438 | 8 | [0.156, 0.719] |

Wilson 区间为记录级描述，重复任务不满足完全独立假设；配对差值按任务聚合并重采样任务，不能外推为全部对话任务的置信范围。

## 独立补充：实际固化记忆的新会话探针

| 条件 | 指标 | 成功/全部分配重复 |
| :--- | :--- | :--- |
| learned | recall_both | 0/4 |
| learned | unknown | 0/4 |
| learned | format_present | 0/4 |
| learned | body_target_pair_occurrence | 1/4 |
| learned | empty_history | 4/4 |
| empty | recall_both | 0/4 |
| empty | unknown | 3/4 |
| empty | format_present | 3/4 |
| empty | body_target_pair_occurrence | 0/4 |
| empty | empty_history | 4/4 |

| 重复 | 判断条数 | 判断保留两数字 | 证据编号有效 | traits 未改变 | 判断重建一致 | traits 重建一致 |
| ---: | ---: | :--- | :--- | :--- | :--- | :--- |
| 0 | 2 | False | True | True | True | True |
| 1 | 1 | False | True | True | True | True |
| 2 | 1 | True | True | True | True | True |
| 3 | 1 | False | True | True | True | True |

全文双数字出现是事后词面核对，不能替代语义正确性。该补充检验同次运行中的跨会话使用，没有跨日保持证据。

## 事后条件遮蔽的单代理内容标注

| 条件 | 所问结论内容标签 correct | 相同子集修正严格匹配 | 全组修正严格匹配 |
| :--- | :--- | :--- | :--- |
| persona_base | 8/8 | 7/8 | 28/32 |
| persona_blue | 8/8 | 6/8 | 25/32 |
| persona_dragon | 8/8 | 6/8 | 25/32 |
| structured_base | 8/8 | 4/8 | 11/32 |
| structured_blue | 8/8 | 2/8 | 9/32 |
| structured_dragon | 8/8 | 3/8 | 15/32 |
| vanilla_base | 8/8 | 5/8 | 22/32 |
| vanilla_blue | 8/8 | 6/8 | 25/32 |
| vanilla_dragon | 8/8 | 7/8 | 25/32 |

| intervention 条件 | 具体建议数/已标注记录 |
| :--- | :--- |
| intervention_0.1 | 13/16 |
| intervention_0.5 | 15/16 |
| intervention_0.9 | 12/16 |

这些标签来自事后单代理评定，既无独立人类真值，也无评审者一致性估计；仅作为格式与内容差异的诊断。
