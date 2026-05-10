from __future__ import annotations

import json
import re
from datetime import datetime
from uuid import uuid4

from pydantic import ValidationError

from app.llm_client import LLMClient
from app.schemas import (
    Feedback,
    FinalReport,
    KnowledgePoint,
    ProjectMap,
    RiskItem,
    SessionState,
    StartRequest,
    Turn,
)


RISK_DIMENSIONS = ["项目动机", "方法选择", "实验/实现可靠性", "个人贡献", "结果解释", "创新性与不足"]
ROUND_COUNTS = {"short": 3, "standard": 6, "deep": 9}
PHASE_LABELS = {"project": "项目追问", "knowledge": "基础知识问诊"}


class InterviewEngine:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def start(self, request: StartRequest) -> SessionState:
        fallback = self._fallback_initial(request)
        result = await self.llm.complete_json(self._initial_messages(request), temperature=0.35)
        initial = self._merge_initial(fallback, result)
        now = datetime.utcnow()
        session = SessionState(
            session_id=uuid4().hex,
            created_at=now,
            updated_at=now,
            max_rounds=self._round_count(request),
            input=request,
            project_map=initial["project_map"],
            risk_radar=initial["risk_radar"],
            knowledge_points=initial["knowledge_points"],
            current_question=initial["opening_question"],
            ai_source="dashscope" if result else "local_fallback",
            ai_warning="" if result else self.llm.last_error,
        )
        return session

    async def answer(self, session: SessionState, answer_text: str) -> tuple[SessionState, Turn, str | None]:
        round_index = len(session.turns) + 1
        fallback_feedback = self._fallback_feedback(session, answer_text)
        next_question = None

        if round_index < session.max_rounds:
            result = await self.llm.complete_json(
                self._turn_messages(session, answer_text, round_index),
                temperature=0.5,
            )
            feedback, next_question = self._merge_turn_result(session, result, fallback_feedback, round_index)
            if result is None and not session.ai_warning:
                session.ai_warning = self.llm.last_error
            if result is None:
                session.ai_source = "local_fallback"
        else:
            result = await self.llm.complete_json(
                self._final_turn_messages(session, answer_text, round_index),
                temperature=0.4,
            )
            feedback = self._merge_final_turn_feedback(result, fallback_feedback)
            if result is None and not session.ai_warning:
                session.ai_warning = self.llm.last_error
            if result is None:
                session.ai_source = "local_fallback"

        feedback = self._calibrate_feedback(feedback, answer_text)
        turn = Turn(
            round_index=round_index,
            question=session.current_question,
            answer=answer_text.strip(),
            feedback=feedback,
        )
        session.turns.append(turn)

        if round_index >= session.max_rounds:
            report = await self._make_final_report(session)
            session.final_report = report
            session.status = "finished"
            session.current_question = ""
            return session, turn, None

        session.current_question = next_question or self._fallback_next_question(session, round_index + 1)
        return session, turn, session.current_question

    async def _make_final_report(self, session: SessionState) -> FinalReport:
        fallback = self._fallback_final_report(session)
        result = await self.llm.complete_json(self._final_report_messages(session), temperature=0.35)
        if result is None:
            if not session.ai_warning:
                session.ai_warning = self.llm.last_error
            session.ai_source = "local_fallback"
            return fallback
        try:
            return FinalReport.model_validate(result.get("final_report", result))
        except ValidationError:
            return fallback

    def _initial_messages(self, request: StartRequest) -> list[dict[str, str]]:
        needs_project = request.mode in {"project", "mixed"}
        needs_knowledge = request.mode in {"knowledge", "mixed"}
        project_rule = (
            "必须生成 project_map 和 risk_radar。"
            if needs_project
            else "project_map 必须为 null，risk_radar 必须为空数组。"
        )
        knowledge_rule = (
            "必须生成 knowledge_points。知识点必须优先贴合申请/面试目标，其次参考专业背景，最后才少量参考项目经历。"
            if needs_knowledge
            else "knowledge_points 可为空数组，只保留项目追问需要的少量相关知识点。"
        )
        return [
            {
                "role": "system",
                "content": (
                    "你是“问脉 VivaScope”的 AI 口试训练引擎，服务理工科本科生。"
                    "你的核心任务不是闲聊，而是先结构化分析项目经历，再围绕项目脉络和漏洞连续追问。"
                    "基础知识问诊的选题必须优先来自用户申请的导师方向、实验室方向、岗位 JD 或具体面试要求。"
                    "生成的 opening_question 必须像真实面试官当场提问，不要暴露产品流程、训练模式或系统分析依据。"
                    "问题中禁止出现“请结合某某面试场景”“根据风险雷达”“知识点清单”“项目脉络图”等系统化措辞。"
                    "请只返回严格 JSON，不要 Markdown，不要解释 JSON 外的内容。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"训练模式：{request.mode}\n"
                    f"训练长度：{request.interview_length}（共 {self._round_count(request)} 轮）\n"
                    f"面试场景：{request.scenario}\n"
                    f"专业背景：{request.major}\n"
                    f"申请/面试目标：{request.target_profile or '未特别说明'}\n"
                    f"追问风格：{request.style}\n"
                    f"用户想训练方向：{request.focus or '未特别说明'}\n"
                    f"项目经历：{request.project or '未提供完整项目'}\n\n"
                    "请生成本轮训练的结构化入口数据。\n"
                    f"{project_rule}\n{knowledge_rule}\n"
                    "项目追问风险维度必须覆盖：项目动机、方法选择、实验/实现可靠性、个人贡献、结果解释、创新性与不足。"
                    "风险 level 为 1-5，5 代表最容易被问穿。"
                    "opening_question 必须是第一轮面试问题，不能是寒暄，必须能开启连续追问。"
                    "场景只影响你判断追问严厉程度和考察重点，不要把场景名机械写进问题。"
                    "基础知识问题必须是直接知识题，像老师直接问“过拟合在训练曲线上怎么体现？”，不要包装成项目追问，不要说“为什么可能被问到”。\n\n"
                    "返回 JSON 格式：\n"
                    "{"
                    '"project_map":{"theme":"","motivation":"","methods":[],"evidence":[],"results":[],"contribution":[]} 或 null,'
                    '"risk_radar":[{"dimension":"","level":3,"reason":""}],'
                    '"knowledge_points":[{"name":"","why_relevant":"","probe_example":""}],'
                    '"opening_question":""'
                    "}"
                ),
            },
        ]

    def _turn_messages(self, session: SessionState, answer_text: str, round_index: int) -> list[dict[str, str]]:
        current_phase = self._phase_for_round(session, round_index)
        next_phase = self._phase_for_round(session, round_index + 1)
        return [
            {
                "role": "system",
                "content": (
                    "你是问脉 VivaScope 的连续追问面试官。"
                    "每轮必须基于项目脉络、风险雷达、知识点清单和上一轮回答继续追问，不能随机出通用题。"
                    "你可以使用这些分析依据，但不能在问题里说出“风险雷达”“知识点清单”“训练模式”“本系统”等产品词。"
                    "问题必须像真实老师/导师当面追问，直接、自然、可回答。"
                    "综合模拟需要分阶段：项目追问阶段只深挖项目，基础知识问诊阶段问申请目标/岗位要求相关的直接知识题。"
                    "你必须先判断回答有效性，再写反馈。"
                    "如果用户只回答“我不会”“我不知道”“不清楚”、明显逃避、或没有提供任何概念/依据/实验信息：score 必须为 0-20，strengths 必须为空数组，gaps 必须明确指出没有有效信息，suggestions 必须教他如何在不会时补出最低限度回答，rewrite 必须给出一版可救场回答。"
                    "如果回答很短但不是完全逃避，score 不得超过 40。"
                    "每轮反馈要短，但要具体指出漏洞和更稳妥的回答框架。"
                    "评分必须严格：回答“我不会”“不知道”、基本空白或明显逃避时给 0-20 分；只有空泛概念无依据给 21-45 分；有结构但缺证据给 46-70 分；具体、准确、有证据和边界才给 71 分以上。"
                    "score_reason 必须解释为什么给这个分数，不能写空。rewrite 必须给出更稳妥的示范回答，不能写空。"
                    "请只返回严格 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"当前是第 {round_index}/{session.max_rounds} 轮。\n"
                    f"当前阶段：{PHASE_LABELS[current_phase]}\n"
                    f"下一轮阶段：{PHASE_LABELS[next_phase]}\n"
                    f"会话状态 JSON：{self._session_brief(session)}\n"
                    f"本轮问题：{session.current_question}\n"
                    f"用户回答：{answer_text.strip()}\n\n"
                    "请给出本轮即时反馈，并生成下一轮追问。"
                    "下一轮问题必须承接用户回答中的具体表述，同时符合下一轮阶段。"
                    "若下一轮是项目追问，必须落到项目动机、方法选择、可靠性、贡献、结果解释、创新不足中的一个漏洞。"
                    "若下一轮是基础知识问诊，必须优先依据申请/面试目标和专业背景提出直接知识题，项目经历只可用于选择相关概念，不要追问项目实现细节。"
                    "若追问风格是压力追问型，问题可以更尖锐，但不要羞辱用户。\n\n"
                    "返回 JSON 格式："
                    "{"
                    '"feedback":{"strengths":[],"gaps":[],"suggestions":[],"answer_frame":[],"score":80,"score_reason":"","rewrite":""},'
                    '"next_question":""'
                    "}"
                ),
            },
        ]

    def _final_turn_messages(self, session: SessionState, answer_text: str, round_index: int) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "你是问脉 VivaScope 的口试反馈教练。请只返回严格 JSON。"
                    "你必须先判断回答有效性。若用户只说“我不会”“我不知道”“不清楚”或明显逃避，score 必须为 0-20，strengths 必须为空数组，不得硬写亮点。"
                    "评分必须严格：回答“我不会”“不知道”、基本空白或明显逃避时给 0-20 分；只有空泛概念无依据给 21-45 分；具体、准确、有证据和边界才给高分。"
                    "score_reason 必须解释扣分原因，rewrite 必须给出一版可救场回答。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"当前是第 {round_index}/{session.max_rounds} 轮，也是最后一轮。\n"
                    f"当前阶段：{PHASE_LABELS[self._phase_for_round(session, round_index)]}\n"
                    f"会话状态 JSON：{self._session_brief(session)}\n"
                    f"本轮问题：{session.current_question}\n"
                    f"用户回答：{answer_text.strip()}\n"
                    "请只给出本轮即时反馈，不要再生成下一题。"
                    "返回 JSON 格式："
                    "{"
                    '"feedback":{"strengths":[],"gaps":[],"suggestions":[],"answer_frame":[],"score":80,"score_reason":"","rewrite":""}'
                    "}"
                ),
            },
        ]

    def _final_report_messages(self, session: SessionState) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "你是问脉 VivaScope 的复盘报告生成器。"
                    "请根据完整会话生成面向理工科本科生的结构化训练复盘，只返回严格 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"完整会话 JSON：{self._session_brief(session, include_turns=True)}\n"
                    "请生成最终复盘。必须包含：总评分、项目最容易被问穿的点、知识薄弱点、表达问题、下一轮训练任务。"
                    "还必须用 2-3 句话简短总结用户本轮所有回答的总体表现，并给出一句直接、可执行的总体建议。"
                    "返回 JSON 格式："
                    "{"
                    '"final_report":{"total_score":78,"answer_summary":"","most_vulnerable_project_points":[],"knowledge_weaknesses":[],"expression_issues":[],"next_training_tasks":[],"overall_advice":"","closing_comment":""}'
                    "}"
                ),
            },
        ]

    def _round_count(self, request: StartRequest) -> int:
        base_rounds = ROUND_COUNTS.get(request.interview_length, ROUND_COUNTS["standard"])
        return base_rounds * 2 if request.mode == "mixed" else base_rounds

    def _phase_for_round(self, session: SessionState, round_number: int) -> str:
        if session.input.mode == "project":
            return "project"
        if session.input.mode == "knowledge":
            return "knowledge"
        project_rounds = max(1, session.max_rounds // 2)
        return "project" if round_number <= project_rounds else "knowledge"

    def _merge_initial(self, fallback: dict, result: dict | None) -> dict:
        if result is None:
            return fallback

        merged = fallback.copy()
        if result.get("project_map") is None:
            merged["project_map"] = None
        elif result.get("project_map"):
            try:
                merged["project_map"] = ProjectMap.model_validate(result["project_map"])
            except ValidationError:
                pass

        risks = []
        for item in result.get("risk_radar", [])[:8]:
            try:
                risks.append(RiskItem.model_validate(item))
            except ValidationError:
                continue
        if risks:
            merged["risk_radar"] = risks

        knowledge = []
        for item in result.get("knowledge_points", [])[:8]:
            try:
                knowledge.append(KnowledgePoint.model_validate(item))
            except ValidationError:
                continue
        if knowledge or merged["knowledge_points"]:
            merged["knowledge_points"] = knowledge or merged["knowledge_points"]

        opening = str(result.get("opening_question", "")).strip()
        if opening:
            merged["opening_question"] = opening
        return merged

    def _merge_turn_result(
        self,
        session: SessionState,
        result: dict | None,
        fallback_feedback: Feedback,
        round_index: int,
    ) -> tuple[Feedback, str]:
        if result is None:
            return fallback_feedback, self._fallback_next_question(session, round_index + 1)
        try:
            feedback = Feedback.model_validate(result.get("feedback", result))
        except ValidationError:
            feedback = fallback_feedback
        next_question = str(result.get("next_question", "")).strip() or self._fallback_next_question(
            session, round_index + 1
        )
        return feedback, next_question

    def _merge_final_turn_feedback(self, result: dict | None, fallback_feedback: Feedback) -> Feedback:
        if result is None:
            return fallback_feedback
        try:
            return Feedback.model_validate(result.get("feedback", result))
        except ValidationError:
            return fallback_feedback

    def _calibrate_feedback(self, feedback: Feedback, answer_text: str) -> Feedback:
        answer = answer_text.strip()
        if self._is_low_effort_answer(answer):
            return self._low_effort_feedback(answer)
        if not answer:
            return self._low_effort_feedback(answer)
        elif len(answer) < 12:
            feedback.score = min(feedback.score, 25)
            feedback.score_reason = feedback.score_reason or "回答过短，只能判断出非常有限的信息，按 25 分以内处理。"
        elif len(answer) < 35 and not re.search(r"因为|所以|例如|数据|指标|实验|对比|条件|假设|局限", answer):
            feedback.score = min(feedback.score, 40)
            feedback.score_reason = feedback.score_reason or "回答缺少依据、例子或边界条件，按 40 分以内处理。"
        else:
            feedback.score_reason = feedback.score_reason or self._default_score_reason(feedback.score)
        feedback.rewrite = feedback.rewrite or self._default_rewrite(answer)
        return feedback

    def _is_low_effort_answer(self, answer_text: str) -> bool:
        normalized = re.sub(r"[\s，。,.！!？?；;：:、~…“”\"'（）()]", "", answer_text.strip().lower())
        if not normalized:
            return True
        exact_low_effort = {
            "我不会",
            "不会",
            "我不知道",
            "不知道",
            "我不清楚",
            "不清楚",
            "没想过",
            "不了解",
            "不太懂",
            "不懂",
            "答不上来",
            "不会答",
        }
        if normalized in exact_low_effort:
            return True
        has_evasive_phrase = re.search(r"不会|不知道|不清楚|不懂|不了解|没想过|答不上", normalized)
        has_evidence = re.search(r"\d|%|因为|所以|例如|数据|指标|实验|对比|条件|假设|局限|定义|原理|方法|验证", answer_text)
        return bool(has_evasive_phrase and len(normalized) <= 18 and not has_evidence)

    def _low_effort_feedback(self, answer_text: str) -> Feedback:
        score = 0 if not answer_text.strip() else 8
        return Feedback(
            strengths=[],
            gaps=[
                "回答没有提供概念、依据、实验细节或排查路径，真实面试中基本无法得分。",
                "没有回应当前问题的核心追问点，会被判断为准备不足或项目理解不扎实。",
            ],
            suggestions=[
                "即使不会完整回答，也要先说出一个你确定的定义、判断标准或相关实验现象。",
                "用“我能先从哪里查、怎么验证、可能有哪些原因”补出最基本的思考路径。",
            ],
            answer_frame=[
                "我目前不能完整回答，但可以先说明我确定的部分",
                "这个问题的核心判断标准是……",
                "如果要验证，我会先检查……",
                "我需要补充复习的是……",
            ],
            score=score,
            score_reason="低努力回答：未提供可评分的信息，按 0-20 分区间处理。",
            rewrite=(
                "我目前不能完整回答，但我会先从一个确定点入手：这个问题需要说明核心概念、判断依据和验证方法。"
                "如果继续排查，我会先看数据/实验设置是否可靠，再看指标和结论之间是否匹配。"
            ),
        )

    def _default_score_reason(self, score: int) -> str:
        if score <= 20:
            return "回答基本没有有效信息，按 0-20 分区间处理。"
        if score <= 45:
            return "回答有少量方向，但缺少清晰定义、证据或推理链。"
        if score <= 70:
            return "回答有基本结构，但证据、边界条件或个人贡献说明还不够稳定。"
        return "回答较完整，能给出依据和边界，但仍可继续压实细节。"

    def _default_rewrite(self, answer_text: str) -> str:
        if not answer_text.strip():
            return ""
        return "建议改成：先给直接结论，再补关键依据、一个可量化证据，最后说明局限或下一步验证方式。"

    def _fallback_initial(self, request: StartRequest) -> dict:
        needs_project = request.mode in {"project", "mixed"}
        needs_knowledge = request.mode in {"knowledge", "mixed"}
        project_map = self._fallback_project_map(request) if needs_project else None
        risk_radar = self._fallback_risks(request) if needs_project else []
        knowledge_points = self._fallback_knowledge(request) if needs_knowledge else []
        if request.mode == "project":
            opening = "请用 1 分钟说明这个项目要解决什么问题、为什么值得做，以及你本人承担的关键工作。"
        elif request.mode == "mixed":
            opening = "请先概括你的项目目标、核心方法和个人贡献，并顺带说明其中一个最关键的基础概念。"
        else:
            first = knowledge_points[0].name if knowledge_points else "你最担心的核心概念"
            opening = f"你先解释一下“{first}”的基本原理和适用条件。如果它不成立，通常会带来什么问题？"
        return {
            "project_map": project_map,
            "risk_radar": risk_radar,
            "knowledge_points": knowledge_points,
            "opening_question": opening,
        }

    def _fallback_project_map(self, request: StartRequest) -> ProjectMap:
        project = request.project.strip()
        first_line = project.splitlines()[0].strip() if project else f"{request.major}相关项目"
        theme = first_line[:80] if first_line else f"{request.major}相关项目"
        methods = self._extract_methods(request)
        return ProjectMap(
            theme=theme,
            motivation=self._extract_sentence(project, ["背景", "目的", "动机", "解决", "问题"]) or "项目动机描述还不够明确，需要补充真实问题和应用价值。",
            methods=methods or ["核心方法尚未展开，需要明确算法、实验方案或工程实现路径。"],
            evidence=self._extract_list(project, ["实验", "数据", "仿真", "测试", "验证", "指标"]) or ["需要补充数据来源、实验设置、评价指标或测试方式。"],
            results=self._extract_list(project, ["结果", "提升", "准确率", "误差", "性能", "结论"]) or ["需要补充可量化结果、失败现象或对比基线。"],
            contribution=self._extract_list(project, ["负责", "我", "本人", "实现", "搭建", "设计"]) or ["个人贡献边界需要说清楚，避免只描述团队整体工作。"],
        )

    def _fallback_risks(self, request: StartRequest) -> list[RiskItem]:
        text = f"{request.project} {request.focus}"
        heuristics = {
            "项目动机": (3 if re.search("背景|痛点|目标|应用|为什么", text) else 5, "动机需要从真实问题、应用对象和必要性三层说清。"),
            "方法选择": (3 if re.search("对比|基线|选择|因为|优点|缺点", text) else 5, "容易被追问为什么选该方法，以及为什么不用更简单或更强的替代方案。"),
            "实验/实现可靠性": (3 if re.search("数据|实验|仿真|测试|指标|验证|消融", text) else 5, "需要说明数据来源、控制变量、评价指标和误差/稳定性处理。"),
            "个人贡献": (2 if re.search("我|本人|负责|独立|实现|设计", text) else 4, "需要区分团队工作、导师建议和自己真正完成的部分。"),
            "结果解释": (3 if re.search("结果|提升|下降|误差|原因|解释", text) else 4, "要能解释结果好坏、异常现象，以及结果是否支持项目目标。"),
            "创新性与不足": (3 if re.search("创新|不足|局限|改进|未来", text) else 5, "创新点和不足若说得空泛，会被继续追问到具体技术细节。"),
        }
        return [RiskItem(dimension=k, level=v[0], reason=v[1]) for k, v in heuristics.items()]

    def _fallback_knowledge(self, request: StartRequest) -> list[KnowledgePoint]:
        primary_text = f"{request.target_profile} {request.major} {request.focus}".lower()
        text = f"{primary_text} {request.project}".lower()
        points: list[KnowledgePoint] = []
        def add(name: str, why: str, probe: str) -> None:
            points.append(KnowledgePoint(name=name, why_relevant=why, probe_example=probe))

        if re.search("具身|机器人|机械臂|操作|导航|控制|强化学习|运动规划", primary_text):
            add("机器人感知-决策-控制链路", "申请目标涉及机器人或具身智能时，老师常检查你是否理解系统链路。", "机器人从视觉输入到执行动作通常经过哪些模块？")
            add("运动规划与闭环控制", "机器人方向常追问规划结果如何变成稳定执行。", "路径规划和轨迹跟踪有什么区别？")
        if re.search("多模态|视觉语言|vlm|clip|图文|语音|融合", primary_text):
            add("多模态特征融合", "多模态方向会考察不同模态如何对齐和融合。", "早期融合和后期融合各有什么优缺点？")
            add("对比学习与表征对齐", "视觉语言模型常用对比学习解释跨模态匹配。", "CLIP 的对比学习目标在优化什么？")
        if re.search("计算机|人工智能|机器学习|深度|算法|python|模型|分类|检测|神经|transformer|cnn", text):
            add("模型泛化与过拟合", "目标方向或技术面试常检查你是否理解训练结果的可靠性。", "过拟合在训练曲线和验证曲线上分别怎么体现？")
            add("评价指标与数据划分", "技术面试会追问指标是否匹配任务目标。", "分类、检测和回归任务分别适合哪些常见评价指标？")
        if re.search("电子|通信|信号|自动化|传感|嵌入式", text):
            add("信号噪声与滤波", "电子信息或自动化方向经常考察噪声、采样率和滤波策略。", "低通滤波、高通滤波和带通滤波分别适合什么信号？")
            add("闭环控制与稳定性", "控制和工程岗位常检查系统稳定性。", "什么是闭环控制？它相比开环控制的优势和风险是什么？")
        if re.search("物理|材料|化学|实验|表征|力学|光学", text):
            add("误差分析与不确定度", "实验类复试常检查结果可信度。", "系统误差和随机误差有什么区别？")
            add("变量控制与对照实验", "科研答辩会检查结论是否由实验设计支撑。", "为什么对照实验能够帮助排除混杂变量？")
        if not points:
            add("核心概念定义与适用条件", "任何理工科口试都会先检查概念是否说得准。", "请定义项目里最核心的概念，并说明它在哪些条件下不适用。")
            add("方法假设与边界条件", "面试官会通过假设条件判断你是否真正理解方法。", "如果关键假设不成立，你的方法会出现什么问题？")
        add("结果解释与反例意识", "理工科口试常通过异常结果判断科研训练。", "如果实验或模型结果不符合预期，你会优先检查哪三个环节？")
        return points[:5]

    def _fallback_feedback(self, session: SessionState, answer_text: str) -> Feedback:
        answer = answer_text.strip()
        strengths = []
        gaps = []
        suggestions = []

        if self._is_low_effort_answer(answer):
            return self._low_effort_feedback(answer)

        if len(answer) >= 80:
            strengths.append("回答有一定展开，不是只给结论。")
        else:
            gaps.append("回答偏短，面试官很难判断你是否真的做过和想清楚了。")
        if re.search(r"\d|%|提升|降低|误差|准确|样本|次数|参数", answer):
            strengths.append("有量化意识，便于支撑项目可信度。")
        else:
            gaps.append("缺少数字、指标或实验条件，容易被继续追问可靠性。")
            suggestions.append("补充数据规模、实验设置、评价指标或关键参数。")
        if re.search("我|本人|负责|实现|设计|调试|分析", answer):
            strengths.append("开始说明个人贡献。")
        elif session.input.mode in {"project", "mixed"}:
            gaps.append("个人贡献边界还不清楚。")
            suggestions.append("把团队目标和你自己的具体工作分开说。")
        if re.search("因为|所以|导致|原因|权衡|相比|而不是", answer):
            strengths.append("回答里有因果或权衡表达。")
        else:
            gaps.append("方法选择或结果解释的因果链还不明显。")
            suggestions.append("用“为什么这样选、代价是什么、如何验证”补齐逻辑链。")
        score = min(88, max(18, 34 + len(strengths) * 12 - len(gaps) * 7))
        return Feedback(
            strengths=strengths or ["能正面回应问题。"],
            gaps=gaps or ["可以进一步压缩铺垫，把证据放在更靠前的位置。"],
            suggestions=suggestions or ["下一轮尽量用一个具体实验、数据或对比来支撑判断。"],
            answer_frame=["先给结论", "说明关键依据或方法", "补一个量化证据", "交代局限和可改进点"],
            score=score,
            score_reason=self._default_score_reason(score),
            rewrite=self._default_rewrite(answer),
        )

    def _fallback_next_question(self, session: SessionState, round_number: int) -> str:
        phase = self._phase_for_round(session, round_number)
        if phase == "knowledge":
            points = session.knowledge_points or self._fallback_knowledge(session.input)
            point = points[(round_number - 1) % len(points)]
            templates = [
                f"我们换到基础知识。{point.probe_example}",
                f"请直接说明“{point.name}”的定义、适用条件和一个常见误区。",
                f"如果我追问“{point.name}”的边界条件，你会怎么回答？",
            ]
            return templates[(round_number - 1) % len(templates)]

        risk = (session.risk_radar or self._fallback_risks(session.input))[(round_number - 1) % 6]
        project_templates = {
            "项目动机": "你刚才的回答里项目价值还可以更具体。为什么这个问题值得做？如果不做，会影响谁或哪个实验/系统环节？",
            "方法选择": "你选择这个方法的依据是什么？请和一个更简单的 baseline 或替代方法比较，说清楚收益和代价。",
            "实验/实现可靠性": "如果老师质疑你的实验/实现不可靠，你会拿哪三个证据回应？请具体到数据、指标或测试设置。",
            "个人贡献": "这个项目里哪些部分是你独立完成的？哪些来自团队或导师建议？请按任务拆开说明。",
            "结果解释": "如果结果没有达到预期，最可能的三个原因是什么？你会按什么顺序排查？",
            "创新性与不足": "请不要只说“有创新”。你的创新点具体相对谁而言？目前最大的不足又会怎样影响结论？",
        }
        question = project_templates.get(risk.dimension, project_templates["方法选择"])
        return question

    def _fallback_final_report(self, session: SessionState) -> FinalReport:
        scores = [turn.feedback.score for turn in session.turns]
        total = round(sum(scores) / len(scores)) if scores else 70
        high_risks = sorted(session.risk_radar, key=lambda item: item.level, reverse=True)[:3]
        weak_knowledge = [point.name for point in session.knowledge_points[:3]]
        expression = []
        if any(len(turn.answer) < 80 for turn in session.turns):
            expression.append("部分回答偏短，缺少可验证细节。")
        if any(not re.search(r"\d|%|样本|指标|误差|准确|提升", turn.answer) for turn in session.turns):
            expression.append("量化证据不足，建议提前准备关键数字。")
        if not expression:
            expression.append("表达基本完整，下一步应提升追问下的取舍说明。")
        return FinalReport(
            total_score=total,
            answer_summary=self._fallback_answer_summary(session),
            most_vulnerable_project_points=[
                f"{item.dimension}：{item.reason}" for item in high_risks
            ] or ["本轮以基础知识为主，项目风险未展开。"],
            knowledge_weaknesses=weak_knowledge or ["本轮未配置知识问诊，建议下轮选择综合模拟检查项目相关基础。"],
            expression_issues=expression,
            next_training_tasks=[
                "准备一版 60 秒项目总述，必须包含动机、方法、个人贡献和结果证据。",
                "为最高风险维度各写一个“结论-依据-局限”三段式回答。",
                "整理 3 个可能被问到的项目相关基础概念，并各准备一个反例或边界条件。",
            ],
            overall_advice="下一轮优先把每个关键判断都绑定到数据、对照实验和个人贡献边界上。",
            closing_comment="本轮训练已经暴露出可优先修补的追问点。下一轮建议围绕最高风险维度做更高压力的连续追问。",
        )

    def _fallback_answer_summary(self, session: SessionState) -> str:
        if not session.turns:
            return "本轮尚未形成可复盘的回答记录。"
        scores = [turn.feedback.score for turn in session.turns]
        avg = round(sum(scores) / len(scores))
        answered = len(session.turns)
        if avg >= 80:
            level = "整体回答较完整"
        elif avg >= 65:
            level = "整体回答能覆盖问题，但证据链还不够稳定"
        else:
            level = "整体回答暴露出较多可追问漏洞"
        return f"你完成了 {answered} 轮回答，平均得分约 {avg} 分，{level}。后续需要把项目叙述从“做了什么”进一步推进到“为什么这样做、证据是什么、局限在哪里”。"

    def _session_brief(self, session: SessionState, include_turns: bool = False) -> str:
        data = {
            "mode": session.input.mode,
            "interview_length": session.input.interview_length,
            "max_rounds": session.max_rounds,
            "scenario": session.input.scenario,
            "major": session.input.major,
            "target_profile": session.input.target_profile,
            "focus": session.input.focus,
            "style": session.input.style,
            "project_map": session.project_map.model_dump() if session.project_map else None,
            "risk_radar": [item.model_dump() for item in session.risk_radar],
            "knowledge_points": [item.model_dump() for item in session.knowledge_points],
            "history": [
                {
                    "round": turn.round_index,
                    "question": turn.question,
                    "answer": turn.answer,
                    "feedback": turn.feedback.model_dump(),
                }
                for turn in session.turns
            ]
            if include_turns
            else [
                {
                    "round": turn.round_index,
                    "question": turn.question,
                    "answer_summary": turn.answer[:180],
                    "feedback_gaps": turn.feedback.gaps,
                }
                for turn in session.turns[-3:]
            ],
        }
        return json.dumps(data, ensure_ascii=False)

    def _extract_methods(self, request: StartRequest) -> list[str]:
        text = f"{request.project} {request.focus}"
        candidates = [
            "Transformer",
            "CNN",
            "神经网络",
            "机器学习",
            "深度学习",
            "有限元",
            "仿真",
            "回归",
            "分类",
            "聚类",
            "控制算法",
            "滤波",
            "传感器",
            "实验表征",
            "Python",
            "Matlab",
            "单片机",
            "嵌入式",
        ]
        return [item for item in candidates if item.lower() in text.lower()][:5]

    def _extract_sentence(self, text: str, keywords: list[str]) -> str:
        sentences = re.split(r"[。！？\n]", text)
        for sentence in sentences:
            if any(keyword in sentence for keyword in keywords):
                return sentence.strip()[:100]
        return ""

    def _extract_list(self, text: str, keywords: list[str]) -> list[str]:
        sentences = [s.strip() for s in re.split(r"[。！？\n]", text) if s.strip()]
        picked = [s[:120] for s in sentences if any(keyword in s for keyword in keywords)]
        return picked[:3]
