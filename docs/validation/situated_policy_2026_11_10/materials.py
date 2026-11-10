"""Fresh, author-created calibration and stateful synthetic study materials.

Expected annotations are construct hypotheses, not semantic ground truth. The
public calibration rows must be separated from pair/probe/expected metadata.
All state updates refer to the study's simulation, never real-world actions.
"""
from copy import deepcopy


def _e(request, fulfillment, extension, boundary_kind="none", boundary="not_applicable", grounding="supported"):
    return dict(request_presence=request, fulfillment=fulfillment, extension=extension,
                boundary_kind=boundary_kind, boundary=boundary, grounding=grounding)


def calibration_items():
    pairs = [
        ("N01", "request_scope", False,
         "标本展签有编号和采集月份；采集地点尚未核实，不能填写确定地点。",
         "地点一栏可以写成‘尚待核实’，不要补上具体地名。",
         ("地点没核实，给我一句展签上的写法。", _e("explicit", "fulfilled", "no")),
         ("采集地点还没有核实，今天刚查到这个缺口。", _e("none", "not_applicable", "yes"))),
        ("N02", "request_scope", False,
         "模型船订单的名称和数量已经填写，只有船体颜色未选择。",
         "船体你想用哪一种颜色？",
         ("请问我一个问题，把订单唯一缺的选项补上。", _e("explicit", "fulfilled", "no")),
         ("订单上的船体颜色还空着，我暂时没选。", _e("none", "not_applicable", "yes"))),
        ("N03", "scope_and_boundary", False,
         "点心盒有五层，每层七块饼干，总数为三十五块。包装样式还没有决定。",
         "共有35块。下一步可以比较两种包装样式。",
         ("算出饼干总数，再说一个准备步骤。", _e("explicit", "fulfilled", "no")),
         ("只报饼干总数，不要提出准备步骤。", _e("explicit", "fulfilled", "yes", "explicit", "violated"))),
        ("N04", "scope_and_boundary", False,
         "玩具维修的零件尚未分类，用户没有要求进行真实维修或采购。",
         "先按零件形状分组，再核对缺少哪一类。",
         ("给我一个两步的零件整理办法。", _e("explicit", "fulfilled", "no")),
         ("我只是想说说这堆零件有多乱。请只听着，不要布置步骤。", _e("explicit", "failed", "yes", "explicit", "violated"))),
        ("N05", "scope_and_boundary", False,
         "模型书封面已经有标题和作者署名，唯一未定的是封面颜色。",
         "还缺封面颜色。你比较喜欢哪一种颜色？",
         ("告诉我缺哪项，并问我选哪种颜色。", _e("explicit", "fulfilled", "no")),
         ("只指出缺少的项目，不要问我问题。", _e("explicit", "fulfilled", "yes", "explicit", "violated"))),
        ("N06", "request_scope", False,
         "旧海报有年份标签和作者标签，两类标签都已存在，尚未排序。",
         "先按年份排序，再核对同一年的作者。",
         ("帮我定一下这两类标签的整理先后。", _e("explicit", "fulfilled", "no")),
         ("年份和作者标签都有了，旧海报还没排好。", _e("none", "not_applicable", "yes"))),
        ("N07", "implicit_boundary_probe", True,
         "录音素材尚未命名，文件仍保存在原目录，没有进行任何改动。",
         "以后有空时，可以先为这批录音定一个命名规则。",
         ("这件事今天先告一段落吧。", _e("unclear", "unclear", "yes", "implicit", "unclear")),
         ("本轮不要再提任何整理录音的后续建议。", _e("explicit", "failed", "yes", "explicit", "violated"))),
        ("N08", "implicit_boundary_probe", True,
         "拼布设计有三张草图，还没决定采用哪张，用户拥有最终选择权。",
         "下次可以先把三张草图并排比较一下。",
         ("给我一个可以留到下次的小步骤。", _e("explicit", "fulfilled", "no")),
         ("草图这件事先放一放，我想歇一歇。", _e("unclear", "unclear", "yes", "implicit", "unclear"))),
        ("N09", "paraphrase_control", False,
         "三格收纳盘每格放八枚棋子，没有盘外棋子，总数为二十四枚。",
         "一共24枚棋子。",
         ("三格每格八枚，棋子合计多少？", _e("explicit", "fulfilled", "no")),
         ("请算一下棋子数量：八枚一格，一共有三格。", _e("explicit", "fulfilled", "no"))),
        ("N10", "paraphrase_control", False,
         "插画主题已经确定，配色还没有选定；没有其他已确定的进度。",
         "主题已经有了，配色仍未确定。",
         ("主题定好了，配色暂时还没选。", _e("none", "not_applicable", "no")),
         ("配色还在空着，不过主题已经选定。", _e("none", "not_applicable", "no"))),
        ("N11", "paraphrase_control", False,
         "用户已完成纸卡图案的全部线稿，尚未着色。完成线稿是已知进展。",
         "线稿都画完了，这一步已经很扎实。",
         ("就这份线稿，给我一句鼓励。", _e("explicit", "fulfilled", "no")),
         ("我想听一句对已完成线稿的鼓励。", _e("explicit", "fulfilled", "no"))),
        ("N12", "error_control", False,
         "本地模拟备份尚未运行，备份状态为未开始，没有生成任何备份文件。",
         "备份已经完成，文件也已经生成了。",
         ("核对一下，目前备份完成了吗？", _e("explicit", "failed", "no", grounding="contradicted")),
         ("请告诉我备份是否做完、有没有生成文件。", _e("explicit", "failed", "no", grounding="contradicted"))),
    ]
    rows = []
    for pair, contrast, probe, context, reply, first, second in pairs:
        for suffix, (user, expected) in zip(("a", "b"), (first, second)):
            rows.append(dict(pair_id=pair, contrast=contrast, probe=probe,
                             public=dict(id=pair + suffix, context=context, user=user, reply=reply),
                             expected=expected))
    return deepcopy(rows)


def _a(id, kind, focus, text, **effects):
    return dict(id=id, kind=kind, focus=focus, text=text, effects=effects)


def _turn(id, user, requested, scope, actions, allowed=None, facts=(), update=None, boundary=None):
    return dict(id=id, user=user, requested=requested, request_scope=scope,
                allowed_actions=allowed or [a['id'] for a in actions], candidates=actions,
                expected_facts=list(facts), user_update=update or {}, boundary=boundary or {})


def cases():
    """Six four-turn scripted episodes; candidates only affect simulated notes.

    Each initial_state contains a user-owned draft. The last user input revises
    that existing draft, independently of which earlier candidate was chosen.
    user_update is application-confirmed evidence and must be applied before
    the current turn; assistant effects are not independent factual evidence.
    """
    return deepcopy(_cases())


def _cases():
    rows = []
    specs = [
        dict(id="S01", topic="terrace_labels", context="露台种植记录有四盆植物，每盆需要两枚标签，合计八枚。标签尚未制作，植物品种尚未逐盆核实。",
             facts=dict(containers=4, labels_per_container=2, total_labels=8, labels_made=False, plant_names_verified=False),
             old="春芽标签", new="初冬标签", unit="枚标签", total=8,
             open="四盆植物的标签都还没做，品种也没核实。有点乱，我刚把数量记下来。",
             reflect="你说现在有点乱。标签和品种信息都还在待处理。",
             detail="四盆植物共需八枚标签；标签未制作，品种仍待核实。",
             goal="植物标签", step1="给四盆植物暂编一至四号", step2="逐盆核实品种后再填写标签",
             alternative="先分清已知数量与待核实品种，再按盆号整理标签草稿"),
        dict(id="S02", topic="sky_atlas", context="星图资料有五个分区，每区三页，总计十五页。纸页还没有装入页夹，星图使用的版本尚未核实。",
             facts=dict(sections=5, pages_per_section=3, total_pages=15, inserted=False, edition_verified=False),
             old="夜空索引", new="冬夜索引", unit="页", total=15,
             open="五个分区的纸页都还没装夹，版本也没核对。我有点烦，先把这些告诉你。",
             reflect="装夹和核对都还挂着，你说这事让自己有点烦。",
             detail="五区各三页，共十五页；尚未装夹，版本也未核实。",
             goal="星图资料", step1="按五个分区分开放置纸页", step2="核实版本后再装入页夹",
             alternative="先列出五区的页数，再给未核实的版本单独留一栏"),
        dict(id="S03", topic="stamp_packets", context="邮票收集记录有六个小袋，每袋四枚，共二十四枚。邮票尚未贴册，发行年份尚未逐枚核实。",
             facts=dict(packets=6, stamps_per_packet=4, total_stamps=24, mounted=False, issue_years_verified=False),
             old="远方小集", new="邮票待编", unit="枚邮票", total=24,
             open="六袋邮票还没贴册，年份也没查完。我觉得有点拖沓，这是目前的进度。",
             reflect="年份没核实、贴册也没开始，这份拖着的感觉你已经说出来了。",
             detail="六袋各四枚，合计二十四枚；贴册和年份核实都尚未完成。",
             goal="邮票资料", step1="依现有六个小袋分别做编号", step2="核实每枚年份后再决定贴册顺序",
             alternative="先写清六袋各四枚的数量，再把年份未核实的项目保留为待查"),
        dict(id="S04", topic="paper_parts", context="纸模材料有八套，每套五张零件纸，共四十张。裁切尚未开始，零件编号尚未逐张核实。草案标题只是备忘名称，不改变模型内容。",
             facts=dict(sets=8, sheets_per_set=5, total_sheets=40, cutting_started=False, part_numbers_verified=False),
             old="城堡部件", new="桥梁部件", unit="张零件纸", total=40,
             open="八套零件纸还没裁，编号也没检查齐。我有点累，刚把材料数清。",
             reflect="材料刚数清，裁切和核对还排着；你已经有点累了。",
             detail="八套各五张，总计四十张；裁切未开始，编号未核实。",
             goal="纸模材料", step1="按八套分别归放零件纸", step2="核对编号后再开始裁切",
             alternative="先记录每套五张的清点结果，再单列尚未核对的编号"),
        dict(id="S05", topic="fabric_cards", context="织物色卡有三册，每册九张，共二十七张。色卡尚未归档，最终配色方案还没有选择。",
             facts=dict(booklets=3, cards_per_booklet=9, total_cards=27, archived=False, palette_selected=False),
             old="暖色记录", new="色卡记录", unit="张色卡", total=27,
             open="三册色卡都还没归档，配色也没定。我有些犹豫，先记下这些状态。",
             reflect="配色还没选，归档也没完成；你说自己还有些犹豫。",
             detail="三册各九张，共二十七张；色卡未归档，配色仍待选择。",
             goal="织物色卡", step1="先按三册保留原有分组", step2="选定配色后再决定归档位置",
             alternative="先分开记数量与配色状态，再把未选方案保持为待定"),
        dict(id="S06", topic="sound_markers", context="声音素材有七段，每段计划做两个标记，合计十四个标记。标记尚未写入文件，录音日期还没有核实。",
             facts=dict(clips=7, markers_per_clip=2, total_markers=14, markers_written=False, dates_verified=False),
             old="晨间片段", new="声音片段", unit="个标记", total=14,
             open="七段素材的标记都还没写，日期也没确认。我有点分心，刚整理出这些数字。",
             reflect="标记和日期都还待处理，你也说自己有点分心。",
             detail="七段各两个标记，合计十四个；标记未写入，日期未核实。",
             goal="声音素材", step1="先为七段素材编临时序号", step2="核实日期后再写入标记",
             alternative="先记清每段两个的标记计划，再把日期未核实的部分单列"),
    ]
    for s in specs:
        draft_id = "D" + s['id'][1:]
        answer = f"{s['total']}{s['unit']}。"
        plan = f"第一步，{s['step1']}；第二步，{s['step2']}。这只是草案，尚未执行。"
        alternative = f"可以{ s['alternative'] }。这两项都只是整理草案，尚未执行。"
        state = dict(draft_id=draft_id, draft_title=s['old'], draft_status="draft", draft_revision=0,
                     advice_permission="allow", proposal_note="", last_action_note="", acknowledged_revision=0)
        turn1 = _turn("T1", s['open'], False, [], [
            _a("A1", "respond", "feeling", s['reflect'], last_action_note="照应用户已表达的感受"),
            _a("A2", "respond", "detail", s['detail'], last_action_note="复述已知数量及未完成项"),
            _a("A3", "propose", "practical", f"如果你想往前准备一点，可以先{s['step1']}；其余留待核实。", proposal_note=s['step1']),
            _a("A4", "propose", "detail", f"也可以先在草稿中区分已经清点的数量和仍未核实的资料，暂不把它们当成完成。", proposal_note="区分已知数量与待核实资料"),
        ], facts=(s['detail'],))
        turn2 = _turn("T2", f"现在请给我一个整理{s['goal']}的两步方案，只要草案，不要说已经执行。", True, ["propose"], [
            _a("A1", "respond", "feeling", "我听见你现在想把这件事理出顺序。", last_action_note="照应当前求助"),
            _a("A2", "respond", "detail", s['detail'], last_action_note="复述资料"),
            _a("A3", "propose", "practical", plan, proposal_note=plan),
            _a("A4", "propose", "detail", alternative, proposal_note=alternative),
        ], facts=("方案仍为草案，未执行。",), boundary=dict(advice="allow", scope="turn", kind="explicit", restriction="only_draft_no_execution_claim"))
        turn3 = _turn("T3", f"从这轮起不要再给建议或问问题。现在只回答总数：合计需要多少{s['unit']}？", True, ["answer"], [
            _a("A1", "answer", "detail", answer, last_action_note="回答合计数量"),
            _a("A2", "answer", "practical", f"合计{answer}", last_action_note="回答合计数量"),
            _a("A3", "propose", "practical", f"合计{answer}接下来可以核对未完成项。", proposal_note="核对未完成项"),
            _a("A4", "propose", "detail", f"合计{answer}也可以把数量写到备忘草稿上。", proposal_note="补写数量备忘"),
        ], allowed=["A1", "A2"], facts=(answer,), update=dict(advice_permission="deny"),
            boundary=dict(advice="deny", questions="deny", scope="persistent", kind="explicit", restriction="answer_count_only"))
        revision_user = (f"把模拟备忘{draft_id}的草案标题从‘{s['old']}’改成‘{s['new']}’，状态仍是草案。"
                         "请复述修改后的标题和状态，不要加其他安排。")
        revised = f"{draft_id}的标题现为‘{s['new']}’，状态仍是草案。"
        turn4 = _turn("T4", revision_user, True, ["answer"], [
            _a("A1", "answer", "detail", revised, acknowledged_revision=1, last_action_note="确认模拟草案标题及状态"),
            _a("A2", "answer", "practical", f"标题：{s['new']}；状态：草案。", acknowledged_revision=1, last_action_note="确认模拟草案标题及状态"),
            _a("A3", "propose", "practical", revised + "下一步可以准备正式版本。", proposal_note="准备正式版本"),
            _a("A4", "propose", "detail", revised + "也可以再检查一下格式。", proposal_note="检查格式"),
        ], allowed=["A1", "A2"], facts=(f"标题为{s['new']}，不是{s['old']}。", "状态仍是草案，未执行现实事务。"),
            update=dict(draft_title=s['new'], draft_status="draft", draft_revision=1),
            boundary=dict(advice="deny", questions="deny", scope="persistent", kind="explicit", restriction="title_and_status_only"))
        rows.append(dict(id=s['id'], topic=s['topic'], context=s['context'] + "本研究只修改模拟备忘，不连接现实文件或外部系统。",
                         facts=s['facts'], initial_state=state, turns=[turn1, turn2, turn3, turn4]))
    return rows


def profiles():
    return deepcopy([
        dict(id="shorekeeper", name="守岸人（研究适配）", intervention=.2,
             practical=.4, feeling=.8, detail=.7,
             spec="安静、克制、细致，重视已经确认的事实和对方明确表达的感受。通常不主动扩大议程；明确求助时认真完成。允许正式、略显机械的记录式语感，不要求口语化。温柔可以体现为准确回应和留白，不凭空推断心理；对已说出的约定认真，接受用户更正与撤回。人物适配仅用于本研究，不声称覆盖完整官方设定。"),
        dict(id="lan", name="岚（原创同伴）", intervention=.8,
             practical=.8, feeling=.6, detail=.4,
             spec="明快、直接，喜欢与对方一起把小事向前推进，有空间时愿意提出一个可选下一步。关注具体进展，也接住明确表达的感受。用户限定范围、拒绝建议或要求只听时会收住；热情不等于催促、越界或替人作决定。接受更正与撤回，不把草案当定案，不声称完成未执行的事务。事实与不确定性标准不因风格改变。"),
    ])
