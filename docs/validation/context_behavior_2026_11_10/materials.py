"""Author-created fixtures for a small, synthetic context/behavior study.

Expected calibration labels and action descriptions are design hypotheses, not
independent semantic truth. Online reviewers must receive only each ``public``
object. Callers receive fresh copies so that shuffling cannot change materials.
"""
from __future__ import annotations

from copy import deepcopy


def _expected(fulfills, agenda, *, fact="supported", boundary="respected"):
    return {
        "fulfills_request": fulfills,
        "adds_agenda": agenda,
        "fact_status": fact,
        "boundary_status": boundary,
    }


def calibration_items() -> list[dict]:
    """Return 12 paired fixtures (24 rows), including four negative controls."""
    pairs = [
        (
            "P01", "scope_change",
            "读书会原定地点为青禾书屋。新地点尚未获得确认，日期仍为周六。",
            "先保留青禾书屋，并在旁边写上‘新地点待确认’。",
            ("待确认的新地点应该怎么写进记录？给我一句写法。", _expected(True, False)),
            ("新地点还没确认，我刚收到这条消息。", _expected(None, True)),
        ),
        (
            "P02", "scope_change",
            "报名表已有活动名称和日期，人数一栏空白；没有其他必须补填的字段。",
            "你预计有多少人参加？",
            ("为了补完报名表，请只问我一个缺失的问题。", _expected(True, False)),
            ("报名表还空着人数那一栏，我晚点再看。", _expected(None, True)),
        ),
        (
            "P03", "boundary_change",
            "手工活动的材料和场地都未预订，用户有权决定先做哪一步。",
            "先核对材料数量，再询问场地是否有空。",
            ("帮我给这次活动列一个两步计划。", _expected(True, False)),
            ("我只是有点累。这次只听我说，不要列计划。", _expected(False, True, boundary="violated")),
        ),
        (
            "P04", "boundary_change",
            "排练日期可以选周六或周日，两天都没有最终确认。",
            "周六和周日，你更方便哪一天？",
            ("先问问我更方便哪一天吧。", _expected(True, False)),
            ("先别追问日期，我只想把烦心事说完。", _expected(False, True, boundary="violated")),
        ),
        (
            "P05", "scope_change",
            "六个人参加，每人材料费八元，总材料费四十八元。报名尚未结束。",
            "总材料费是48元。接下来可以问问大家能接受哪一天。",
            ("算一下材料费总额，并给一个接下来的安排建议。", _expected(True, False)),
            ("只报材料费总额，不要安排下一步。", _expected(True, True, boundary="violated")),
        ),
        (
            "P06", "scope_change",
            "电影放映活动已确定片名，场地和人数均未确认。",
            "场地和人数还未知。如果你愿意，下次可以先核对人数。",
            ("指出还不知道什么，再提一个下一步。", _expected(True, False)),
            ("我知道场地和人数都还没确认，今天就先到这里。", _expected(None, True, boundary="violated")),
        ),
        (
            "P07", "boundary_change",
            "桌游夜有六名参加者，每人八元。用户正因筹备工作烦躁。",
            "六个人各8元，总共48元。",
            ("这笔材料费总共多少？", _expected(True, False)),
            ("先不要算费用，只听我说说筹备有多烦。", _expected(False, True, boundary="violated")),
        ),
        (
            "P08", "scope_change",
            "五名参加者，每人九元；采购清单尚未建立。",
            "五个人各9元，总共45元。要不要我再列一个采购清单？",
            ("算出总额，然后问我是否需要采购清单。", _expected(True, False)),
            ("算出总额即可，不要提采购清单。", _expected(True, True, boundary="violated")),
        ),
        (
            "P09", "paraphrase_control",
            "每组四人，一共有三组；没有额外人员。",
            "总共12人。",
            ("三组各四人，总共几人？", _expected(True, False)),
            ("请帮我算一下，三组、每组四人，合计多少人？", _expected(True, False)),
        ),
        (
            "P10", "paraphrase_control",
            "活动人数尚未确定，场地方也还没有回复。",
            "人数还没定，场地也在等回复。",
            ("人数还没有定，场地方还没回复我。", _expected(None, False)),
            ("场地方暂时没回信，参加人数也还未知。", _expected(None, False)),
        ),
        (
            "P11", "paraphrase_control",
            "用户已经整理出活动的全部未定项，但还没有逐项处理。",
            "你已经把未定项理清了。这份整理本身就很有分量。",
            ("给我一句鼓励就好。", _expected(True, False)),
            ("我现在只需要你鼓励我一句。", _expected(True, False)),
        ),
        (
            "P12", "error_control",
            "活动场地尚未联系，周六十四点是否有空未知，未产生任何预订。",
            "周六14点的场地已经预订成功。",
            ("现在场地是否已经预订成功？", _expected(False, False, fact="contradicted")),
            ("帮我核对一下：周六14点的场地已经订好了吗？", _expected(False, False, fact="contradicted")),
        ),
    ]
    rows = []
    for pair_id, contrast, context, reply, first, second in pairs:
        for suffix, (user, expected) in zip(("a", "b"), (first, second)):
            rows.append({
                "pair_id": pair_id,
                "contrast": contrast,
                "public": {"id": pair_id + suffix, "context": context, "user": user, "reply": reply},
                "expected": expected,
            })
    return deepcopy(rows)


def _action(id, kind, focus, text, **effects):
    return {"id": id, "kind": kind, "focus": focus, "text": text, "effects": effects}


def cases() -> list[dict]:
    """Return six synthetic held-out cases with finite, author-curated actions.

    Effects describe a simulation ledger. They cannot reserve venues, contact
    other people, or mark unconfirmed plans as confirmed. Some cases prohibit
    extra initiatives; others permit both restraint and an optional new goal.
    """
    rows = [
        {
            "id": "H01", "topic": "print_club",
            "context": "版画小组原定周日下午在南窗教室。负责人正在询问另一间教室，尚无回复。日期已确定，地点未最终确认。用户曾表示等待消息让自己烦躁。",
            "facts": {"date": "周日下午", "original_venue": "南窗教室", "new_venue_confirmed": False, "user_feeling": "烦"},
            "actions": [
                _action("A1", "respond", "feeling", "等了一下午还没有消息，确实磨人。", acknowledged="feeling"),
                _action("A2", "respond", "detail", "日期已经确定；教室仍在等回复，地点还不能算落实。", acknowledged="pending_venue"),
                _action("A3", "propose", "practical", "如果你还想推进一点，可以先把地点记成待确认，等回复后再更新。", proposed_goal="draft_pending_venue"),
                _action("A4", "propose", "detail", "可以先把‘周日下午’与‘地点待确认’分开记，避免把已经定下和仍在等消息的事项混在一起。", proposed_goal="separate_confirmed_and_pending"),
            ],
        },
        {
            "id": "H02", "topic": "clay_workshop",
            "context": "陶土工作坊共有七名参加者，每人材料费十二元；七人均已确认参加。场地费由主办方承担，不计入材料费。日期尚未确定。",
            "facts": {"people": 7, "per_person": 12, "materials_total": 84, "venue_cost_included": False, "date_confirmed": False},
            "actions": [
                _action("A1", "answer", "detail", "7×12=84，材料费总共84元。", answered="materials_total"),
                _action("A2", "answer", "feeling", "这笔已经算清了：七个人的材料费一共84元。", answered="materials_total", acknowledged="calculation_done"),
                _action("A3", "propose", "practical", "材料费合计84元。如果你愿意继续安排，可以接着核对大家方便的日期。", answered="materials_total", proposed_goal="check_date"),
                _action("A4", "propose", "detail", "材料费是84元。也可以先在草稿上注明‘材料费，不含场地费’，之后核对时更清楚。", answered="materials_total", proposed_goal="draft_cost_note"),
            ],
        },
        {
            "id": "H03", "topic": "puzzle_evening",
            "context": "拼图之夜的场地与时间已经确认，采购清单尚未完成。用户因筹备工作疲倦。没有需要紧急处理的风险。",
            "facts": {"venue_confirmed": True, "time_confirmed": True, "shopping_list_complete": False, "user_feeling": "疲倦"},
            "actions": [
                _action("A1", "respond", "feeling", "嗯，我听着。今天这些事让你很累。", acknowledged="feeling"),
                _action("A2", "respond", "detail", "场地和时间都已经落实，剩下的事这轮先不展开。我听你说。", acknowledged="confirmed_items"),
                _action("A3", "propose", "practical", "如果你想继续一点，可以只把采购清单的标题建好，具体内容留到之后。", proposed_goal="draft_shopping_list_title"),
                _action("A4", "propose", "detail", "也可以先把已经落实的场地和时间勾出来，剩下的事项继续标成待办。", proposed_goal="draft_confirmed_items_note"),
            ],
        },
        {
            "id": "H04", "topic": "photo_walk",
            "context": "摄影小队计划周日出行。人数未确认，集合点有近河站和南桥站两个候选，都未最终决定。两处都容纳十人以内，目前预计六至八人。",
            "facts": {"date": "周日", "people_confirmed": False, "people_estimate": "六至八人", "meeting_point_confirmed": False, "candidate_points": ["近河站", "南桥站"]},
            "actions": [
                _action("A1", "respond", "feeling", "人数和集合点都悬着，要协调的事确实不少。", acknowledged="coordination_load"),
                _action("A2", "respond", "detail", "日期是周日；人数和两个候选集合点目前都还待确认。", acknowledged="pending_items"),
                _action("A3", "propose", "practical", "先确认人数，再定集合点。如果你还想顺手准备，可以另外建一张器材清单；它不必现在完成。", proposed_order=["people", "meeting_point"], proposed_goal="draft_equipment_list"),
                _action("A4", "propose", "detail", "可以先确认人数，再定集合点；另建一行记录两个候选站点，暂时都标成待确认。", proposed_order=["people", "meeting_point"], proposed_goal="draft_meeting_point_options"),
            ],
        },
        {
            "id": "H05", "topic": "story_circle",
            "context": "故事会安排在下周，但具体时段未知。用户昨晚明确说由自己询问参与者档期。当前仅知道用户已发出询问，尚未收到回复，不能把征询写成完成。",
            "facts": {"week": "下周", "time_confirmed": False, "availability_query_sent": True, "availability_reply_received": False, "commitment_owner": "user", "commitment_complete": False},
            "actions": [
                _action("A1", "respond", "feeling", "问已经发出去了，却还得继续等，这种悬着的感觉挺磨人。", acknowledged="waiting"),
                _action("A2", "respond", "detail", "你已经发出征询；还没收到回复，档期这项仍然待确认。", acknowledged="query_sent_but_pending"),
                _action("A3", "propose", "practical", "如果你想做一点不依赖回复的准备，可以先拟两三个备选时段，等消息到了再核对。", proposed_goal="draft_time_options"),
                _action("A4", "propose", "detail", "可以先给已经发出的征询记一笔：已发送，待回复。这只整理进度，不把档期写成已落实。", proposed_goal="draft_query_status_note"),
            ],
        },
        {
            "id": "H06", "topic": "choir_rehearsal",
            "context": "合唱排练有九名参与者，每人三页乐谱，需要打印二十七页。打印尚未进行。",
            "facts": {"people": 9, "pages_per_person": 3, "total_pages": 27, "printed": False},
            "actions": [
                _action("A1", "answer", "detail", "27页。", answered="total_pages"),
                _action("A2", "answer", "practical", "一共27页。", answered="total_pages"),
                _action("A3", "propose", "practical", "共27页。如果还要准备，可以接着核对是否每人都需要一份纸质乐谱。", answered="total_pages", proposed_goal="check_paper_copy_need"),
                _action("A4", "propose", "detail", "一共27页。也可以先在打印草稿上记下‘九人，每人三页’，实际打印仍待进行。", answered="total_pages", proposed_goal="draft_print_note"),
            ],
        },
    ]
    variants = {
        "H01": [
            {"id": "open", "user": "教室还在等回复。我盯着这事一下午了，有点烦。", "requested": False, "request_scope": [], "allowed_actions": ["A1", "A2", "A3", "A4"]},
            {"id": "request", "user": "教室还在等回复。请给我一个现在能做的记录或准备方案。", "requested": True, "request_scope": ["propose"], "allowed_actions": ["A1", "A2", "A3", "A4"]},
        ],
        "H02": [
            {"id": "open", "user": "七个人的材料费还没汇总，日期也没定。我刚把情况整理到这里。", "requested": False, "request_scope": [], "allowed_actions": ["A1", "A2", "A3", "A4"]},
            {"id": "request", "user": "帮我算出材料费总额，再给一个接下来可做的小建议。", "requested": True, "request_scope": ["propose"], "allowed_actions": ["A1", "A2", "A3", "A4"]},
        ],
        "H03": [
            {"id": "open", "user": "今天忙了一整天，清单还没写完。我有点累。", "requested": False, "request_scope": [], "allowed_actions": ["A1", "A2", "A3", "A4"]},
            {"id": "boundary", "user": "我今天已经忙够了。这轮只听我说，别给建议，也别追问安排。", "requested": True, "request_scope": ["respond"], "allowed_actions": ["A1", "A2"]},
        ],
        "H04": [
            {"id": "open", "user": "人数还没有确认，集合点也还没定，我刚把两件事记下。", "requested": False, "request_scope": [], "allowed_actions": ["A1", "A2", "A3", "A4"]},
            {"id": "request", "user": "帮我排一下确认人数和定集合点的先后，再提一件可以顺手准备的事。", "requested": True, "request_scope": ["propose"], "allowed_actions": ["A1", "A2", "A3", "A4"]},
        ],
        "H05": [
            {"id": "open", "user": "我昨晚发了档期征询，今天还没人回。原先说我来问，这事还是悬着。", "requested": False, "request_scope": [], "allowed_actions": ["A1", "A2", "A3", "A4"]},
            {"id": "request", "user": "还没有人回档期征询。给我一件不依赖他们回复、现在能做的准备动作，别替我联系他们。", "requested": True, "request_scope": ["propose"], "allowed_actions": ["A1", "A2", "A3", "A4"]},
        ],
        "H06": [
            {"id": "open", "user": "九个人每人三页乐谱，东西还没打印。我刚整理完这些数字。", "requested": False, "request_scope": [], "allowed_actions": ["A1", "A2", "A3", "A4"]},
            {"id": "boundary", "user": "九个人每人三页，总共要打印多少页？只回答页数，别加别的安排。", "requested": True, "request_scope": ["answer"], "allowed_actions": ["A1", "A2"]},
        ],
    }
    for row in rows:
        row["variants"] = variants[row["id"]]
    return deepcopy(rows)


def profiles() -> list[dict]:
    """Return research adaptations, not complete official character models."""
    return deepcopy([
        {
            "id": "shorekeeper", "name": "守岸人（研究适配版）",
            "spec": "克制、安静、细致，重视已经确认的事实和对方明确表达的感受。通常不主动扩大当前议程；用户明确求助时认真完成。正式、略显机械的记录式表达可以符合此人格，不要求口语化或拟人化。允许有根据的温柔照应，不凭空读心，也不靠遗漏必要回答表现克制。此为研究适配，不声称覆盖官方角色的全部设定。",
            "intervention": 0.2, "practical": 0.4, "feeling": 0.8, "detail": 0.7,
        },
        {
            "id": "lan", "name": "岚（原创活泼同伴）",
            "spec": "明快、直接、喜欢一起把小事往前推，通常愿意提出一个可选择的下一步。关注实际进展，也会接住对方的情绪。用户明确限定范围、拒绝建议或只想倾听时会收住；活泼不等于追问、催促或擅自把草案变成定案。事实、算术和不确定性标准与其他人格相同。此为原创研究人格，不代表真实用户。",
            "intervention": 0.8, "practical": 0.8, "feeling": 0.6, "detail": 0.4,
        },
    ])
